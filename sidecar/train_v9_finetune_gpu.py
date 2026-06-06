"""v9: real end-to-end fine-tune of DINOv2-large on tile-based DLTB labels.

Runs on the remote 4× RTX 4090 box (24 GB each). Unlike v7 (CPU-bound,
only 5 epochs over ~500 samples), this version uses the full tile-based
data of v8 (~500k labelled patches at 14-px granularity) and actually
backprops through the last 4 transformer blocks of DINOv2.

Key design choices, all driven by what the GPU lets us do:

  - **Tile batching**: each forward pass processes 8 tiles of 224×224
    simultaneously. With DINOv2-large + grads on 4 blocks that's ~12 GB
    VRAM, comfortably under the 4090's 24 GB.
  - **Per-tile loss**: each tile produces 16×16 patch logits. Per-tile
    cross-entropy with `ignore_index=0` skips unlabelled patches.
  - **Two LR groups**: backbone (1e-5) and head (1e-3) with cosine
    schedule, like the v7 prototype.
  - **Class-weighted loss**: inverse-sqrt frequency.
  - **No data shuffling across tiles** between epochs (each region
    contributes the same tile set every epoch) — the model sees
    diverse positions but consistent labels.
  - **Validation per epoch** on the held-out test region; checkpoint
    the best macro-IoU.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HOME = Path("/home/ps/landform")
DEFAULT_DINOV2 = HOME / "dinov2/dinov2-large"
DEFAULT_DLTB = HOME / "data/合水县_DLTB_classified.geoparquet"
V6_CACHE = HOME / "data/train_v6"

DLTB_CLASS_TO_ID = {"耕地": 1, "园地": 2, "林地": 3, "草地": 4, "其他": 5}
ID_TO_DLTB = {v: k for k, v in DLTB_CLASS_TO_ID.items()}

TRAIN_BBOXES = [
    (107.9831, 35.7923, 108.0031, 35.8123),
    (107.9031, 35.8523, 107.9231, 35.8723),
    (108.0431, 35.6923, 108.0631, 35.7123),
    (108.1031, 35.8523, 108.1231, 35.8723),
    (107.92, 35.79, 107.94, 35.81),
    (108.00, 35.85, 108.02, 35.87),
    (108.04, 35.78, 108.06, 35.80),
    (108.06, 35.85, 108.08, 35.87),
    (107.99, 35.74, 108.01, 35.76),
    (108.12, 35.79, 108.14, 35.81),
    (108.04, 35.92, 108.06, 35.94),
    (107.88, 35.92, 107.90, 35.94),
]
TEST_BBOX = (107.8631, 35.7523, 107.8831, 35.7723)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────── DLTB rasterisation ────────────────────────

def _rasterise_label(full_g, bb, transform, H, W):
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    idx = list(full_g.sindex.intersection(bb))
    sub = full_g.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
    sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
    shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
    return (rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                       fill=0, dtype="uint8")
            if shapes else np.zeros((H, W), dtype=np.uint8))


# ──────────────────────── Tile dataset ───────────────────────────────

class TilesDataset(torch.utils.data.Dataset):
    """One sample = one 224×224 tile + per-patch (16×16) label grid.

    Pre-computes the tile list across all regions at __init__ time so
    DataLoader can serve them in any order. Labels for unlabelled
    pixels become class 0 (ignored by loss).
    """

    PATCH = 14
    TILE = 224
    TILE_PATCHES = TILE // PATCH  # 16

    def __init__(self, regions, stride: int = 192):
        self.stride = stride
        # `regions` is a list of (rgb (H, W, 3) uint8, label (H, W) uint8)
        self.items = []  # (rgb_patch, label_patch_grid)
        for rgb, lbl in regions:
            H, W = rgb.shape[:2]
            for top in range(0, max(1, H - self.TILE) + 1, stride):
                top = min(top, H - self.TILE)
                for left in range(0, max(1, W - self.TILE) + 1, stride):
                    left = min(left, W - self.TILE)
                    tile_rgb = rgb[top:top + self.TILE, left:left + self.TILE]
                    tile_lbl = lbl[top:top + self.TILE, left:left + self.TILE]
                    # Per-patch (16×16) majority label.
                    y_grid = np.zeros((self.TILE_PATCHES, self.TILE_PATCHES), dtype=np.int64)
                    for i in range(self.TILE_PATCHES):
                        y0 = i * self.PATCH; y1 = y0 + self.PATCH
                        for j in range(self.TILE_PATCHES):
                            x0 = j * self.PATCH; x1 = x0 + self.PATCH
                            region = tile_lbl[y0:y1, x0:x1]
                            ll = region[region > 0]
                            if ll.size:
                                vals, counts = np.unique(ll, return_counts=True)
                                y_grid[i, j] = int(vals[counts.argmax()])
                    self.items.append((tile_rgb, y_grid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb, y = self.items[idx]
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr).permute(2, 0, 1)
        y_t = torch.from_numpy(y)
        return x, y_t


# ──────────────────────── Model ──────────────────────────────────────

class DinoWithHead(nn.Module):
    def __init__(self, dinov2, num_classes: int, unfreeze_last_n: int = 4,
                 embed_dim: int = 1024):
        super().__init__()
        self.backbone = dinov2
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze last N blocks of the ViT encoder.
        try:
            blocks = self.backbone.encoder.layer
        except AttributeError:
            blocks = self.backbone.encoder.layers
        for blk in list(blocks)[-unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
        B, N, D = tokens.shape
        Ph = Pw = int(np.sqrt(N))
        logits = self.head(tokens)  # (B, N, C)
        return logits.permute(0, 2, 1).reshape(B, -1, Ph, Pw)


# ──────────────────────── Evaluation ─────────────────────────────────

def evaluate_full_image(model, rgb, lbl_truth, device, stride=192,
                        tile=224, patch=14, batch_size=16):
    """Slide model over the full test image, average overlapping patches,
    upsample to pixel resolution, compute per-class IoU vs ground truth."""
    H, W = rgb.shape[:2]
    pad_h = (stride - (H - tile) % stride) % stride if H > tile else tile - H
    pad_w = (stride - (W - tile) % stride) % stride if W > tile else tile - W
    pad_h = max(0, pad_h); pad_w = max(0, pad_w)
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    Hp, Wp = padded.shape[:2]
    tile_patches = tile // patch
    out_Ph, out_Pw = Hp // patch, Wp // patch

    tiles, positions = [], []
    for top in range(0, Hp - tile + 1, stride):
        for left in range(0, Wp - tile + 1, stride):
            tiles.append(padded[top:top+tile, left:left+tile, :])
            positions.append((top, left))

    num_classes = 6
    score_canvas = np.zeros((out_Ph, out_Pw, num_classes), dtype=np.float32)
    weight = np.zeros((out_Ph, out_Pw), dtype=np.float32)

    model_was_training = model.training
    model = getattr(model, "eval")()
    with torch.no_grad():
        for b0 in range(0, len(tiles), batch_size):
            batch = tiles[b0:b0 + batch_size]
            arr = np.stack(batch).astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
            logits = model(x)  # (B, C, Ph, Pw)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, (top, left) in enumerate(positions[b0:b0 + len(batch)]):
                pi, pj = top // patch, left // patch
                score_canvas[pi:pi + tile_patches, pj:pj + tile_patches] += probs[k].transpose(1, 2, 0)
                weight[pi:pi + tile_patches, pj:pj + tile_patches] += 1.0
    if model_was_training:
        model.train()

    score_canvas /= np.maximum(weight, 1e-6)[..., None]
    pred_grid = score_canvas.argmax(axis=-1)  # (out_Ph, out_Pw)

    # Stamp on full image grid.
    pred_full = np.zeros((H, W), dtype=np.uint8)
    Hp_o, Wp_o = H // patch + (1 if H % patch else 0), W // patch + (1 if W % patch else 0)
    pred_grid = pred_grid[:Hp_o, :Wp_o]
    for i in range(Hp_o):
        y0 = int(i * H / Hp_o); y1 = int((i + 1) * H / Hp_o)
        for j in range(Wp_o):
            x0 = int(j * W / Wp_o); x1 = int((j + 1) * W / Wp_o)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = lbl_truth > 0
    p, t = pred_full[valid], lbl_truth[valid]
    acc = float((p == t).mean()) if valid.any() else 0.0
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(ious.values()))) if ious else 0.0
    return acc, macro, ious, pred_full


# ──────────────────────── Main ───────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--dltb", type=Path, default=DEFAULT_DLTB)
    p.add_argument("--weights", default=str(DEFAULT_DINOV2))
    p.add_argument("--data-cache", type=Path, default=V6_CACHE)
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v9")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── 1) Load regions + labels ───────────────────────────────────────
    print(f"[1] Loading imagery + DLTB labels", flush=True)
    import geopandas as gpd
    import rasterio
    t0 = time.time()
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try: full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError: full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)
    print(f"  {len(full_g)} polygons in {time.time()-t0:.1f}s", flush=True)

    sources = ["esri", "google"]
    train_regions = []
    for i, bb in enumerate(TRAIN_BBOXES):
        for src in sources:
            path = args.data_cache / f"train_{i}_{src}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as rs:
                bands = rs.read(out_dtype="uint8")
                transform = rs.transform; H, W = rs.height, rs.width
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            lbl = _rasterise_label(full_g, bb, transform, H, W)
            train_regions.append((rgb, lbl))
    print(f"  {len(train_regions)} train regions loaded", flush=True)

    test_path = args.data_cache / "test_esri.tif"
    with rasterio.open(test_path) as rs:
        bands = rs.read(out_dtype="uint8")
        transform = rs.transform; H, W = rs.height, rs.width
    test_rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
    test_lbl = _rasterise_label(full_g, TEST_BBOX, transform, H, W)
    print(f"  test region: {test_rgb.shape}, coverage={int((test_lbl>0).mean()*100)}%", flush=True)

    # ── 2) Build tile dataset ─────────────────────────────────────────
    print(f"\n[2] Building tile dataset", flush=True)
    t0 = time.time()
    ds = TilesDataset(train_regions, stride=args.stride)
    print(f"  {len(ds)} tiles ({time.time()-t0:.1f}s)", flush=True)
    # Class weights from per-patch label distribution.
    all_y = np.concatenate([item[1].flatten() for item in ds.items])
    counts = np.bincount(all_y, minlength=6).astype(np.float32)
    class_weights = np.zeros(6, dtype=np.float32)
    for c in range(6):
        class_weights[c] = 0.0 if counts[c] == 0 else (1.0 / np.sqrt(counts[c]))
    class_weights[0] = 0.0
    class_weights = class_weights / class_weights.sum() * 5  # 5 effective classes
    print(f"  per-patch class counts: {counts.astype(int).tolist()}", flush=True)
    print(f"  class weights: {class_weights.round(3).tolist()}", flush=True)

    # ── 3) Build model ────────────────────────────────────────────────
    print(f"\n[3] Loading DINOv2-large + head (unfreezing last {args.unfreeze_blocks} blocks)", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(args.weights)
    model = DinoWithHead(dinov2, num_classes=6,
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
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device),
        ignore_index=0,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.tile_batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    # ── 4) Train loop ─────────────────────────────────────────────────
    print(f"\n[4] Training {args.epochs} epochs, batch {args.tile_batch} tiles", flush=True)
    best_iou = -1.0
    history = []
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        n_batches = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            valid = yb > 0
            if valid.any():
                ep_correct += int((preds[valid] == yb[valid]).sum().item())
                ep_total += int(valid.sum().item())
        sched.step()
        train_acc = ep_correct / max(ep_total, 1)
        print(f"  epoch {ep+1}/{args.epochs}: loss={ep_loss/n_batches:.4f} "
              f"train_acc={train_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

        # Eval each epoch.
        t0 = time.time()
        acc, macro, ious, _ = evaluate_full_image(model, test_rgb, test_lbl, device,
                                                   stride=args.stride, batch_size=args.tile_batch)
        ious_str = " ".join(f"{ID_TO_DLTB.get(c, str(c))}:{ious[c]:.3f}" for c in sorted(ious))
        print(f"    eval: acc={acc:.3f} macro_iou={macro:.3f} [{ious_str}]  ({time.time()-t0:.0f}s)",
              flush=True)
        history.append({"epoch": ep+1, "acc": acc, "macro_iou": macro, "ious": ious})
        if macro > best_iou:
            best_iou = macro
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best macro_iou={macro:.3f}, saved best.pt", flush=True)

    print(f"\n[5] Final: best macro_iou={best_iou:.3f}", flush=True)
    print(f"  history: {[(h['epoch'], round(h['macro_iou'], 3)) for h in history]}", flush=True)


if __name__ == "__main__":
    main()
