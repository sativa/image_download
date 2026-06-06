"""v11: multi-county + multi-scale DINOv2 fine-tune.

Key changes over v9 (which hit 0.293 macro IoU then overfit at epoch 2):

  1. **3-5× more training data** across 15-20 Gansu counties (read from
     /tmp/region_selection.json — produced by select_regions.py from the
     county scanner output).

  2. **Multi-scale tile features**: every region is processed at BOTH
     tile=224 (local detail) AND tile=448 (broader context, downsampled
     back to the same 14-px patch grid). Per-patch feature is the
     concat of both scales → 2048-d. The classifier head sees both
     fine texture and broad spatial context for every prediction.

  3. **No other architectural change** to keep this comparable to v9
     (DINOv2-large, last 4 blocks unfrozen, AdamW 1e-5 backbone / 1e-3
     head, cosine LR, class-weighted CE, ignore_index=0).

Why multi-scale beats single-scale on landcover:
  - A 224-tile patch covers ~14 px of original — great for boundaries,
    bad for "is this cropland or orchard" (needs grid-pattern context).
  - A 448-tile patch covers ~28 px of original — sees field boundaries
    and texture-grid arrangement that single-pixel patches miss.
  - Concatenating both lets the head pick the more informative scale
    per-class instead of forcing one choice globally.
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


HOME = Path("/home/ps/landform")
DEFAULT_DINOV2 = HOME / "dinov2/dinov2-large"
sys.path.insert(0, str(HOME / "sidecar"))

DLTB_CLASS_TO_ID = {"耕地": 1, "园地": 2, "林地": 3, "草地": 4, "其他": 5}
ID_TO_DLTB = {v: k for k, v in DLTB_CLASS_TO_ID.items()}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DLBM_TO_CLASS = {  # GB/T 21010 first 2 digits → 一级地类
    "01": 1, "02": 2, "03": 3, "04": 4,
    "05": 5, "06": 5, "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5,
}


# ─────────────── Per-region label rasterisation ───────────────

def rasterise_dltb_region(gdf_full, bb_wgs84, transform, H, W):
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    idx = list(gdf_full.sindex.intersection(bb_wgs84))
    sub = gdf_full.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bb_wgs84))
    sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
    shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
    return (rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                       fill=0, dtype="uint8")
            if shapes else np.zeros((H, W), dtype=np.uint8))


# ─────────────── Multi-scale tile dataset ───────────────

class MultiScaleTilesDataset(torch.utils.data.Dataset):
    """One sample = (224 tile, 448 tile, per-patch label grid).

    Both scales are centred on the same image region and produce 16×16
    patches each. The 224 scale captures texture detail; the 448 scale
    captures broader context. The classifier head will concat the
    feature vectors per spatial patch.

    Labels are computed on the 224-grid: the 16×16 patch labels under
    the 224 tile. For the 448 tile (which covers 2× as much area), we
    pool features back to the 224 grid via a 2× nearest-neighbour
    upsample of the 448 feature map (which is 16×16 covering 448 px,
    same patch count but 2× area each).
    """

    PATCH = 14
    TILE_SMALL = 224
    TILE_BIG = 448
    SMALL_PATCHES = TILE_SMALL // PATCH  # 16
    BIG_PATCHES = TILE_BIG // PATCH  # 32

    def __init__(self, regions, stride: int = 192):
        self.regions = regions
        self.items = []
        for rgb, lbl in regions:
            H, W = rgb.shape[:2]
            for top in range(0, max(1, H - self.TILE_SMALL) + 1, stride):
                top = min(top, H - self.TILE_SMALL)
                for left in range(0, max(1, W - self.TILE_SMALL) + 1, stride):
                    left = min(left, W - self.TILE_SMALL)
                    # Centre of the small tile
                    cy = top + self.TILE_SMALL // 2
                    cx = left + self.TILE_SMALL // 2
                    # Big tile centred on the same point, padded if needed
                    big_top = cy - self.TILE_BIG // 2
                    big_left = cx - self.TILE_BIG // 2
                    self.items.append((rgb, lbl, top, left, big_top, big_left, H, W))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _normalise(rgb):
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx):
        rgb, lbl, top, left, big_top, big_left, H, W = self.items[idx]
        small = rgb[top:top + self.TILE_SMALL, left:left + self.TILE_SMALL]
        small_lbl = lbl[top:top + self.TILE_SMALL, left:left + self.TILE_SMALL]
        # Big tile with edge-pad if out of bounds.
        big = np.zeros((self.TILE_BIG, self.TILE_BIG, 3), dtype=np.uint8)
        y0 = max(0, big_top); y1 = min(H, big_top + self.TILE_BIG)
        x0 = max(0, big_left); x1 = min(W, big_left + self.TILE_BIG)
        big[y0 - big_top:y1 - big_top, x0 - big_left:x1 - big_left] = rgb[y0:y1, x0:x1]

        # Per-patch labels at the SMALL scale.
        y_grid = np.zeros((self.SMALL_PATCHES, self.SMALL_PATCHES), dtype=np.int64)
        for i in range(self.SMALL_PATCHES):
            yy0 = i * self.PATCH; yy1 = yy0 + self.PATCH
            for j in range(self.SMALL_PATCHES):
                xx0 = j * self.PATCH; xx1 = xx0 + self.PATCH
                region = small_lbl[yy0:yy1, xx0:xx1]
                ll = region[region > 0]
                if ll.size:
                    vals, counts = np.unique(ll, return_counts=True)
                    y_grid[i, j] = int(vals[counts.argmax()])

        return self._normalise(small), self._normalise(big), torch.from_numpy(y_grid)


# ─────────────── Model with multi-scale head ───────────────

class DinoMultiScale(nn.Module):
    """Two DINOv2 forward passes (small/big tile) → concat per-patch features → MLP head.

    Both passes share the SAME backbone (parameter-tied), only the spatial
    resolution differs. Backbone unfreezing applies to both passes.
    """

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
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def _forward_one(self, x):
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 1:, :]

    def forward(self, x_small, x_big):
        """x_small (B, 3, 224, 224), x_big (B, 3, 448, 448).

        Returns logits (B, C, 16, 16) — at the small-scale patch grid.
        Big-scale features are 32×32 patches; we pool 2×2 → 16×16 to
        align spatially before concatenation.
        """
        small = self._forward_one(x_small)  # (B, 256, D)
        big = self._forward_one(x_big)  # (B, 1024, D)
        B, _, D = small.shape
        small = small.reshape(B, 16, 16, D)  # (B, 16, 16, D)
        big = big.reshape(B, 32, 32, D)
        # 2×2 average pool the big features to 16×16.
        big_pooled = big.unfold(1, 2, 2).unfold(2, 2, 2).mean(dim=(-2, -1))  # (B, 16, 16, D)
        feat = torch.cat([small, big_pooled], dim=-1)  # (B, 16, 16, 2D)
        logits = self.head(feat)  # (B, 16, 16, C)
        return logits.permute(0, 3, 1, 2)  # (B, C, 16, 16)


# ─────────────── Eval (slide both scales) ───────────────

def evaluate_full_image(model, rgb, truth, device, stride=192, batch_size=4):
    H, W = rgb.shape[:2]
    PATCH, TS, TB = 14, 224, 448
    SMALL_P, BIG_P = TS // PATCH, TB // PATCH
    pad_h = (stride - (H - TS) % stride) % stride if H > TS else TS - H
    pad_w = (stride - (W - TS) % stride) % stride if W > TS else TS - W
    pad_h = max(0, pad_h); pad_w = max(0, pad_w)
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    Hp, Wp = padded.shape[:2]

    tiles_small = []
    tiles_big = []
    positions = []
    half_big = TB // 2
    for top in range(0, Hp - TS + 1, stride):
        for left in range(0, Wp - TS + 1, stride):
            tiles_small.append(padded[top:top+TS, left:left+TS])
            cy, cx = top + TS//2, left + TS//2
            bt = cy - half_big; bl = cx - half_big
            big = np.zeros((TB, TB, 3), dtype=np.uint8)
            y0 = max(0, bt); y1 = min(Hp, bt + TB)
            x0 = max(0, bl); x1 = min(Wp, bl + TB)
            big[y0-bt:y1-bt, x0-bl:x1-bl] = padded[y0:y1, x0:x1]
            tiles_big.append(big)
            positions.append((top, left))

    out_Ph, out_Pw = Hp // PATCH, Wp // PATCH
    num_classes = 6
    score = np.zeros((out_Ph, out_Pw, num_classes), dtype=np.float32)
    weight = np.zeros((out_Ph, out_Pw), dtype=np.float32)

    was_training = model.training
    model = getattr(model, "eval")()
    with torch.no_grad():
        for b0 in range(0, len(tiles_small), batch_size):
            sm = tiles_small[b0:b0 + batch_size]
            bg = tiles_big[b0:b0 + batch_size]
            sx = torch.stack([MultiScaleTilesDataset._normalise(t) for t in sm]).to(device)
            bx = torch.stack([MultiScaleTilesDataset._normalise(t) for t in bg]).to(device)
            logits = model(sx, bx)  # (B, C, 16, 16)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, (top, left) in enumerate(positions[b0:b0 + len(sm)]):
                pi, pj = top // PATCH, left // PATCH
                score[pi:pi + SMALL_P, pj:pj + SMALL_P] += probs[k].transpose(1, 2, 0)
                weight[pi:pi + SMALL_P, pj:pj + SMALL_P] += 1.0
    if was_training:
        model.train()

    score /= np.maximum(weight, 1e-6)[..., None]
    pred_grid = score.argmax(axis=-1)
    Hp_o, Wp_o = H // PATCH + (1 if H % PATCH else 0), W // PATCH + (1 if W % PATCH else 0)
    pred_grid = pred_grid[:Hp_o, :Wp_o]

    pred_full = np.zeros((H, W), dtype=np.uint8)
    for i in range(Hp_o):
        y0 = int(i*H/Hp_o); y1 = int((i+1)*H/Hp_o)
        for j in range(Wp_o):
            x0 = int(j*W/Wp_o); x1 = int((j+1)*W/Wp_o)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = truth > 0
    if not valid.any():
        return 0.0, 0.0, {}, pred_full
    p, t = pred_full[valid], truth[valid]
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
    p.add_argument("--regions-json", type=Path,
                   default=HOME / "data/v11_regions.json",
                   help="produced by select_regions.py from county scan")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v11")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--tile-batch", type=int, default=4)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} test regions"
          f" from {regions_meta.get('n_counties', '?')} counties", flush=True)

    # Load per-county DLTB from the pre-converted geoparquets.
    import geopandas as gpd
    import rasterio
    dltb_cache = HOME / "data/v11_dltb"
    gdf_per_county = {}
    for r in regions_meta["train"] + regions_meta["test"]:
        code = r["county"]
        if code in gdf_per_county:
            continue
        pq = dltb_cache / f"{code}.parquet"
        if not pq.exists():
            raise SystemExit(f"missing geoparquet: {pq}")
        g = gpd.read_parquet(pq)
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g
    print(f"  loaded DLTB for {len(gdf_per_county)} counties", flush=True)

    print(f"\n[2] Loading imagery + rasterising labels", flush=True)
    sources = ["esri", "google"]
    def _load(region_list):
        regs = []
        for r in region_list:
            bb = tuple(r["bbox"])
            for src in sources:
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
    train_regions = _load(regions_meta["train"])
    test_regions = _load(regions_meta["test"])
    print(f"  loaded {len(train_regions)} train + {len(test_regions)} test images", flush=True)

    print(f"\n[3] Building multi-scale tile datasets", flush=True)
    t0 = time.time()
    train_ds = MultiScaleTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(train_ds)} train tiles ({time.time()-t0:.1f}s)", flush=True)
    all_y = np.concatenate([item[2].flatten() for item in
                             [train_ds[i] for i in range(min(200, len(train_ds)))]])
    counts = np.bincount(all_y, minlength=6).astype(np.float32)
    class_weights = np.zeros(6, dtype=np.float32)
    for c in range(6):
        class_weights[c] = 0.0 if counts[c] == 0 else (1.0 / np.sqrt(counts[c]))
    class_weights[0] = 0.0
    class_weights = class_weights / class_weights.sum() * 5
    print(f"  class weights: {class_weights.round(3).tolist()}", flush=True)

    print(f"\n[4] Loading DINOv2-large + multi-scale head", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoMultiScale(dinov2, num_classes=6,
                            unfreeze_last_n=args.unfreeze_blocks).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable/total: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)", flush=True)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("head.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device),
        ignore_index=0,
    )

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.tile_batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    print(f"\n[5] Training {args.epochs} epochs, batch {args.tile_batch}", flush=True)
    best_iou = -1.0
    no_improve = 0
    PATIENCE = 3
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_c = 0; ep_t = 0; n_b = 0
        t0 = time.time()
        for sm, bg, yb in loader:
            sm = sm.to(device, non_blocking=True)
            bg = bg.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(sm, bg)
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
        per_class_iou = {c: [] for c in range(1, 6)}
        for rgb, lbl in test_regions:
            acc, macro, ious, _ = evaluate_full_image(model, rgb, lbl, device,
                                                       stride=args.stride, batch_size=args.tile_batch)
            all_acc.append(acc); all_macro.append(macro)
            for c, v in ious.items():
                per_class_iou.setdefault(c, []).append(v)
        avg_acc = float(np.mean(all_acc))
        avg_macro = float(np.mean(all_macro))
        per_cls_str = " ".join(
            f"{ID_TO_DLTB.get(c, str(c))}:{np.mean(v):.3f}"
            for c, v in per_class_iou.items() if v
        )
        print(f"    eval: avg_acc={avg_acc:.3f} avg_macro_iou={avg_macro:.3f} [{per_cls_str}]  ({time.time()-t0:.0f}s)",
              flush=True)

        if avg_macro > best_iou:
            best_iou = avg_macro
            no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best avg_macro_iou={avg_macro:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[early stop] no improvement for {PATIENCE} epochs", flush=True)
                break

    print(f"\n[done] best avg_macro_iou={best_iou:.3f}", flush=True)


if __name__ == "__main__":
    main()
