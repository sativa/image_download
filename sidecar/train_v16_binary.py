"""v16: 广义耕地二分类专项 (broad cropland vs other).

Re-frame the 5-class problem as binary:
  - class 1 = 广义耕地 (originally 耕地 + 园地)
  - class 2 = 非耕地 (originally 林地 + 草地 + 其他)
  - class 0 = unlabelled (ignored)

This collapses the noise in 4 non-cropland classes into one big
"other" bucket, sharpening the model's effective supervision on the
"is this cropland or not" decision boundary.

Same DINOv2-large + UNet decoder + fp16 + composite loss (CE + Lovász
+ Dice). Uses v15's 80-region training set. Reports recall/precision/F1
for the focused cropland-vs-other task. Goal: F1 ≥ 0.80.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import (
    DEFAULT_DINOV2, IMAGENET_MEAN, IMAGENET_STD,
    DLBM_TO_CLASS, DinoUNet,
)
from train_v13_lovasz import lovasz_softmax


def remap_to_binary(lbl):
    """{1, 2} → 1; {3, 4, 5} → 2; 0 → 0 (unchanged)."""
    out = np.zeros_like(lbl)
    out[(lbl == 1) | (lbl == 2)] = 1
    out[(lbl == 3) | (lbl == 4) | (lbl == 5)] = 2
    return out


def rasterise_dltb_binary(gdf, bb_wgs84, transform, H, W):
    """Rasterise DLTB into a binary {1: broad-cropland, 2: other, 0: nodata}."""
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    idx = list(gdf.sindex.intersection(bb_wgs84))
    sub = gdf.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bb_wgs84))
    sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
    # Binary cid: 1 if original is 耕地/园地, else 2.
    sub["bin"] = np.where((sub["cid"] == 1) | (sub["cid"] == 2), 1, 2)
    shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["bin"]) if c > 0]
    return (rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                       fill=0, dtype="uint8")
            if shapes else np.zeros((H, W), dtype=np.uint8))


class BinaryPixelTilesDataset(torch.utils.data.Dataset):
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
        rgb_tile = rgb[top:top+self.TILE, left:left+self.TILE]
        lbl_tile = lbl[top:top+self.TILE, left:left+self.TILE]
        if np.random.random() < 0.5:
            rgb_tile = rgb_tile[:, ::-1, :].copy()
            lbl_tile = lbl_tile[:, ::-1].copy()
        if np.random.random() < 0.5:
            rgb_tile = rgb_tile[::-1, :, :].copy()
            lbl_tile = lbl_tile[::-1, :].copy()
        arr = rgb_tile.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(arr).permute(2, 0, 1),
                torch.from_numpy(lbl_tile.astype(np.int64)))


def composite_binary_loss(logits, labels, cw, ignore: int = 0):
    """Same as v13's but for 3-channel logits (ignore + 2 classes)."""
    logits_fp32 = logits.float()
    ce = F.cross_entropy(logits_fp32, labels, weight=cw, ignore_index=ignore)
    probas = F.softmax(logits, dim=1)
    lov = lovasz_softmax(probas, labels, classes="present", ignore=ignore)
    # Dice on the cropland channel (class 1).
    valid = labels != ignore
    lbl_crop = (labels == 1).float() * valid.float()
    p_crop = probas[:, 1] * valid.float()
    inter = (p_crop * lbl_crop).sum()
    union = p_crop.sum() + lbl_crop.sum()
    dice = 1.0 - (2 * inter + 1e-6) / (union + 1e-6)
    return 0.4 * ce + 0.3 * lov + 0.3 * dice, {
        "ce": ce.item(), "lov": lov.item(), "dice": dice.item(),
    }


def evaluate_full_binary(model, rgb, lbl_truth, device, stride=384, batch_size=8):
    """Slide UNet over image, predict; compute IoU + recall/precision/F1 for class 1."""
    from eval_tta import slide_predict_softmax  # reuse
    # But that one returns 6-channel; we have 3-channel. Custom-roll.
    H, W = rgb.shape[:2]
    TILE = 448
    pad_h = (stride - (H - TILE) % stride) % stride if H > TILE else TILE - H
    pad_w = (stride - (W - TILE) % stride) % stride if W > TILE else TILE - W
    pad_h = max(0, pad_h); pad_w = max(0, pad_w)
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    Hp, Wp = padded.shape[:2]
    score = np.zeros((Hp, Wp, 3), dtype=np.float32)
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
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, (top, left) in enumerate(positions[b0:b0+len(batch)]):
                score[top:top+TILE, left:left+TILE] += probs[k].transpose(1, 2, 0)
                weight[top:top+TILE, left:left+TILE] += 1.0
    if was_training:
        model.train()
    score /= np.maximum(weight, 1e-6)[..., None]
    pred = score.argmax(axis=-1).astype(np.uint8)[:H, :W]

    valid = lbl_truth > 0
    if not valid.any():
        return None
    p, t = pred[valid], lbl_truth[valid]
    crop_p, crop_t = (p == 1), (t == 1)
    tp = int((crop_p & crop_t).sum())
    fp = int((crop_p & ~crop_t).sum())
    fn = int((~crop_p & crop_t).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc = int((p == t).sum()) / int(valid.sum())
    return {"acc": acc, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v15_regions.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v16")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} test (BINARY)", flush=True)

    import geopandas as gpd, rasterio
    gdf_per_county = {}
    for r in regions_meta["train"] + regions_meta["test"]:
        code = r["county"]
        if code in gdf_per_county:
            continue
        g = gpd.read_parquet(args.dltb_cache / f"{code}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g
    print(f"  loaded {len(gdf_per_county)} counties", flush=True)

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
                lbl = rasterise_dltb_binary(gdf_per_county[r["county"]], bb, transform, H, W)
                regs.append((rgb, lbl))
        return regs

    print(f"\n[2] Loading + rasterising binary labels", flush=True)
    t0 = time.time()
    train_regions = _load(regions_meta["train"])
    test_regions = _load(regions_meta["test"])
    print(f"  loaded {len(train_regions)} train + {len(test_regions)} test ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[3] Pixel dataset", flush=True)
    train_ds = BinaryPixelTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(train_ds)} tiles", flush=True)
    bin_counts = np.zeros(3, dtype=np.float64)
    for _, lbl in train_regions:
        bin_counts += np.bincount(lbl.ravel(), minlength=3)
    print(f"  pixel counts: {bin_counts.astype(int).tolist()} (cropland={bin_counts[1]/bin_counts[1:].sum()*100:.1f}%)",
          flush=True)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0
    cw = (cw / cw.sum() * 2)
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)

    print(f"\n[4] Building model (3-channel output: ignore + cropland + other)", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=3, unfreeze_last_n=args.unfreeze_blocks).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,}", flush=True)

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
    scaler = torch.amp.GradScaler("cuda")
    cw_t = torch.from_numpy(cw).to(device)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.tile_batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    print(f"\n[5] Training {args.epochs} epochs, fp16 + composite", flush=True)
    best_f1 = -1.0
    no_improve = 0
    PATIENCE = 5
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; ep_c = 0; ep_t = 0; n_b = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss, parts = composite_binary_loss(logits, yb, cw_t)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ep_loss += loss.item(); n_b += 1
            preds = logits.argmax(dim=1)
            valid = yb > 0
            if valid.any():
                ep_c += int((preds[valid] == yb[valid]).sum().item())
                ep_t += int(valid.sum().item())
        sched.step()
        train_acc = ep_c / max(ep_t, 1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        t0 = time.time()
        all_metrics = []
        for rgb, lbl in test_regions:
            m = evaluate_full_binary(model, rgb, lbl, device,
                                       stride=args.stride, batch_size=args.tile_batch)
            if m: all_metrics.append(m)
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        print(f"    eval: acc={avg['acc']:.3f} iou={avg['iou']:.3f} "
              f"prec={avg['precision']:.3f} rec={avg['recall']:.3f} F1={avg['f1']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop]", flush=True)
                break

    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
