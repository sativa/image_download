"""v12: DINOv2 + UNet decoder, per-pixel CE loss.

Breaks the patch-majority physical ceiling of v3-v11 by predicting a
class for **every pixel** in the input tile (448×448 → 448×448 ×
num_classes) instead of one class per 14-px patch.

Architecture:
  Input: 448×448 RGB tile
  Encoder: DINOv2-large (last 4 blocks unfrozen) → (32, 32, 1024) features
  Decoder: 4× transposed-conv upsampling (32 → 64 → 128 → 256 → 512)
    + final bilinear to 448 + 1×1 classifier → (448, 448, num_classes)
  Loss: per-pixel CE with class weights, ignore_index=0

Why this beats patch-level:
  - Training signal: 200,704 pixel labels per tile vs 256 patches.
  - Decoder learns "what pixel belongs where" — captures within-patch
    diversity (e.g. road through forest, hedgerow between fields).
  - Eval matches the actual ground truth resolution.

Compatible with v11's multi-county data set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
DEFAULT_DINOV2 = HOME / "dinov2/dinov2-large"
sys.path.insert(0, str(HOME / "sidecar"))

DLTB_CLASS_TO_ID = {"耕地": 1, "园地": 2, "林地": 3, "草地": 4, "其他": 5}
ID_TO_DLTB = {v: k for k, v in DLTB_CLASS_TO_ID.items()}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DLBM_TO_CLASS = {
    "01": 1, "02": 2, "03": 3, "04": 4,
    "05": 5, "06": 5, "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5,
}


# ─────────────── DLTB rasterisation (per-county geoparquet) ───────────────

def rasterise_dltb_region(gdf, bb_wgs84, transform, H, W):
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    idx = list(gdf.sindex.intersection(bb_wgs84))
    sub = gdf.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bb_wgs84))
    sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
    shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
    return (rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                       fill=0, dtype="uint8")
            if shapes else np.zeros((H, W), dtype=np.uint8))


# ─────────────── Pixel-level tiles dataset ───────────────

class PixelTilesDataset(torch.utils.data.Dataset):
    """Sample = (448-tile RGB, 448-tile pixel label raster)."""

    TILE = 448

    def __init__(self, regions, stride: int = 384):
        self.items = []
        for rgb, lbl in regions:
            H, W = rgb.shape[:2]
            for top in range(0, max(1, H - self.TILE) + 1, stride):
                top = min(top, H - self.TILE)
                for left in range(0, max(1, W - self.TILE) + 1, stride):
                    left = min(left, W - self.TILE)
                    self.items.append((rgb, lbl, top, left))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb, lbl, top, left = self.items[idx]
        rgb_tile = rgb[top:top + self.TILE, left:left + self.TILE]
        lbl_tile = lbl[top:top + self.TILE, left:left + self.TILE]
        # Random h/v flips for augmentation (landcover labels are invariant).
        if np.random.random() < 0.5:
            rgb_tile = rgb_tile[:, ::-1, :].copy()
            lbl_tile = lbl_tile[:, ::-1].copy()
        if np.random.random() < 0.5:
            rgb_tile = rgb_tile[::-1, :, :].copy()
            lbl_tile = lbl_tile[::-1, :].copy()
        arr = rgb_tile.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x, torch.from_numpy(lbl_tile.astype(np.int64))


# ─────────────── DINOv2 + UNet decoder ───────────────

class DinoUNet(nn.Module):
    def __init__(self, dinov2, num_classes: int = 6, unfreeze_last_n: int = 4,
                 embed_dim: int = 1024):
        super().__init__()
        self.backbone = dinov2
        for p in self.backbone.parameters():
            p.requires_grad = False
        try: blocks = self.backbone.encoder.layer
        except AttributeError: blocks = self.backbone.encoder.layers
        for blk in list(blocks)[-unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        # Decoder: upsample from (32, 32) to (448, 448) gradually.
        self.proj = nn.Conv2d(embed_dim, 256, 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True))   # 32→64
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))    # 64→128
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))    # 128→256
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(16), nn.ReLU(inplace=True))    # 256→512
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[:, 1:, :]  # (B, N, D)
        B, N, D = tokens.shape
        Ph = Pw = int(np.sqrt(N))
        feat = tokens.permute(0, 2, 1).reshape(B, D, Ph, Pw)  # (B, D, 32, 32)
        x = self.proj(feat)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        # x is now (B, 16, 512, 512); resize to 448 to match input.
        if x.shape[-1] != 448:
            x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        return self.classifier(x)  # (B, num_classes, 448, 448)


# ─────────────── Eval (slide UNet over full image) ───────────────

def evaluate_full_image(model, rgb, lbl_truth, device, stride=384, batch_size=4):
    H, W = rgb.shape[:2]
    TILE = 448
    pad_h = (stride - (H - TILE) % stride) % stride if H > TILE else TILE - H
    pad_w = (stride - (W - TILE) % stride) % stride if W > TILE else TILE - W
    pad_h = max(0, pad_h); pad_w = max(0, pad_w)
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    Hp, Wp = padded.shape[:2]

    score = np.zeros((Hp, Wp, 6), dtype=np.float32)
    weight = np.zeros((Hp, Wp), dtype=np.float32)
    tiles, positions = [], []
    for top in range(0, Hp - TILE + 1, stride):
        for left in range(0, Wp - TILE + 1, stride):
            tiles.append(padded[top:top+TILE, left:left+TILE])
            positions.append((top, left))

    was_training = model.training
    model = getattr(model, "eval")()
    with torch.no_grad():
        for b0 in range(0, len(tiles), batch_size):
            batch = tiles[b0:b0+batch_size]
            arr = np.stack(batch).astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
            logits = model(x)  # (B, C, 448, 448)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C, 448, 448)
            for k, (top, left) in enumerate(positions[b0:b0+len(batch)]):
                score[top:top+TILE, left:left+TILE] += probs[k].transpose(1, 2, 0)
                weight[top:top+TILE, left:left+TILE] += 1.0
    if was_training:
        model.train()
    score /= np.maximum(weight, 1e-6)[..., None]
    pred_full = score.argmax(axis=-1).astype(np.uint8)[:H, :W]

    valid = lbl_truth > 0
    if not valid.any():
        return 0.0, 0.0, {}, pred_full
    p, t = pred_full[valid], lbl_truth[valid]
    acc = float((p == t).mean())
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(ious.values()))) if ious else 0.0
    return acc, macro, ious, pred_full


# ─────────────── Main ───────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v11_regions.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb",
                   help="per-county geoparquet files {code}.parquet")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v12")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--tile-batch", type=int, default=4)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} test regions", flush=True)

    # Load per-county DLTB geoparquet (much faster than FGDB on remote).
    import geopandas as gpd
    import rasterio
    gdf_per_county = {}
    for r in regions_meta["train"] + regions_meta["test"]:
        code = r["county"]
        if code in gdf_per_county:
            continue
        pq = args.dltb_cache / f"{code}.parquet"
        if not pq.exists():
            raise SystemExit(f"missing {pq}")
        g = gpd.read_parquet(pq)
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g
    print(f"  loaded DLTB for {len(gdf_per_county)} counties", flush=True)

    def _load(region_list):
        regs = []
        for r in region_list:
            bb = tuple(r["bbox"])
            for src in ["esri", "google"]:
                path = args.data_cache / f"{r['county']}_{r['idx']}_{src}.tif"
                if not path.exists():
                    continue
                with rasterio.open(path) as rs:
                    bands = rs.read(out_dtype="uint8")
                    transform = rs.transform; H, W = rs.height, rs.width
                rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
                lbl = rasterise_dltb_region(gdf_per_county[r["county"]], bb,
                                             transform, H, W)
                regs.append((rgb, lbl))
        return regs

    print(f"\n[2] Loading + rasterising imagery", flush=True)
    t0 = time.time()
    train_regions = _load(regions_meta["train"])
    test_regions = _load(regions_meta["test"])
    print(f"  loaded {len(train_regions)} train, {len(test_regions)} test ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[3] Building pixel tile dataset", flush=True)
    t0 = time.time()
    train_ds = PixelTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(train_ds)} tiles ({time.time()-t0:.1f}s)", flush=True)
    # Per-pixel class distribution → class weights.
    bin_counts = np.zeros(6, dtype=np.float64)
    for _, lbl in train_regions:
        b = np.bincount(lbl.ravel(), minlength=6)
        bin_counts += b
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0)
    cw[0] = 0.0
    cw = (cw / cw.sum() * 5).astype(np.float32)
    print(f"  pixel counts: {bin_counts.astype(int).tolist()}", flush=True)
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)

    print(f"\n[4] Loading DINOv2-large + UNet decoder (unfreeze last {args.unfreeze_blocks})", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=6, unfreeze_last_n=args.unfreeze_blocks).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable/total: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)", flush=True)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not (n.startswith("up") or n.startswith("proj")
                                                    or n.startswith("classifier"))]
    head_params = [p for n, p in model.named_parameters()
                   if (n.startswith("up") or n.startswith("proj") or n.startswith("classifier"))]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).to(device), ignore_index=0)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.tile_batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    print(f"\n[5] Training {args.epochs} epochs", flush=True)
    best_iou = -1.0
    no_improve = 0
    PATIENCE = 4
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_c = 0; ep_t = 0; n_b = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item(); n_b += 1
            preds = logits.argmax(dim=1)
            valid = yb > 0
            if valid.any():
                ep_c += int((preds[valid] == yb[valid]).sum().item())
                ep_t += int(valid.sum().item())
        sched.step()
        train_acc = ep_c / max(ep_t, 1)
        print(f"  epoch {ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        t0 = time.time()
        all_acc, all_macro = [], []
        per_class = {c: [] for c in range(1, 6)}
        for rgb, lbl in test_regions:
            acc, macro, ious, _ = evaluate_full_image(model, rgb, lbl, device,
                                                       stride=args.stride, batch_size=args.tile_batch)
            all_acc.append(acc); all_macro.append(macro)
            for c, v in ious.items():
                per_class.setdefault(c, []).append(v)
        avg_acc = float(np.mean(all_acc))
        avg_macro = float(np.mean(all_macro))
        per_cls_str = " ".join(
            f"{ID_TO_DLTB.get(c, str(c))}:{np.mean(v):.3f}"
            for c, v in per_class.items() if v
        )
        print(f"    eval: avg_acc={avg_acc:.3f} avg_macro_iou={avg_macro:.3f} [{per_cls_str}] ({time.time()-t0:.0f}s)",
              flush=True)
        if avg_macro > best_iou:
            best_iou = avg_macro
            no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best avg_macro_iou={avg_macro:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[early stop]", flush=True)
                break

    print(f"\n[done] best avg_macro_iou={best_iou:.3f}", flush=True)


if __name__ == "__main__":
    main()
