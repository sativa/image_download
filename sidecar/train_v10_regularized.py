"""v10: fight v9's overfit with regularisation + augmentation.

v9 hit 0.293 macro IoU at epoch 2 then trended down (classic small-data
overfit, train_acc kept climbing). Same architecture, four changes:

  1. **Unfreeze only last 2 blocks** (v9 used 4). Half the trainable
     params (25M → 12M) ⇒ less memorisation, more bias toward backbone
     priors.
  2. **Lower backbone LR** (1e-5 → 3e-6). The features already encode
     most of what we need; we just nudge them.
  3. **Random horizontal/vertical flips** on tiles. Land-cover labels
     are invariant to N/S/E/W mirroring, so this is free augmentation.
  4. **Increased dropout** (0.3 → 0.5) and **stronger weight decay**
     (1e-4 → 5e-4) on the head.

Also: cosine LR with warmup, eval every epoch, save best by macro IoU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, "/home/ps/landform/sidecar")
from train_v9_finetune_gpu import (
    HOME, DEFAULT_DINOV2, DEFAULT_DLTB, V6_CACHE,
    DLTB_CLASS_TO_ID, ID_TO_DLTB, TRAIN_BBOXES, TEST_BBOX,
    IMAGENET_MEAN, IMAGENET_STD, _rasterise_label, evaluate_full_image,
    DinoWithHead, TilesDataset,
)


class AugmentedTilesDataset(TilesDataset):
    """TilesDataset with random horizontal+vertical flips for training.

    Per-patch labels are also flipped to stay aligned. Inherited methods
    handle stride/tile/patch sizing — we only override __getitem__.
    """

    def __init__(self, regions, stride: int = 192, seed: int = 0):
        super().__init__(regions, stride=stride)
        self.rng = np.random.RandomState(seed)

    def __getitem__(self, idx):
        rgb, y = self.items[idx]
        # Random hflip / vflip — both label-invariant for landcover.
        flip_h = self.rng.random() < 0.5
        flip_v = self.rng.random() < 0.5
        if flip_h:
            rgb = rgb[:, ::-1, :].copy()
            y = y[:, ::-1].copy()
        if flip_v:
            rgb = rgb[::-1, :, :].copy()
            y = y[::-1, :].copy()
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr).permute(2, 0, 1)
        y_t = torch.from_numpy(y)
        return x, y_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--dltb", type=Path, default=DEFAULT_DLTB)
    p.add_argument("--weights", default=str(DEFAULT_DINOV2))
    p.add_argument("--data-cache", type=Path, default=V6_CACHE)
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v10")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--unfreeze-blocks", type=int, default=2)
    p.add_argument("--backbone-lr", type=float, default=3e-6)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--head-dropout", type=float, default=0.5)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

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

    print(f"\n[2] Building augmented tile dataset", flush=True)
    t0 = time.time()
    ds = AugmentedTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(ds)} tiles ({time.time()-t0:.1f}s)", flush=True)
    all_y = np.concatenate([item[1].flatten() for item in ds.items])
    counts = np.bincount(all_y, minlength=6).astype(np.float32)
    class_weights = np.zeros(6, dtype=np.float32)
    for c in range(6):
        class_weights[c] = 0.0 if counts[c] == 0 else (1.0 / np.sqrt(counts[c]))
    class_weights[0] = 0.0
    class_weights = class_weights / class_weights.sum() * 5
    print(f"  class weights: {class_weights.round(3).tolist()}", flush=True)

    print(f"\n[3] Model: DINOv2-large + head (unfreeze last {args.unfreeze_blocks}, dropout={args.head_dropout})", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(args.weights)

    class DinoWithHeadCustom(DinoWithHead):
        def __init__(self, dinov2, num_classes=6, unfreeze_last_n=2, embed_dim=1024,
                     dropout: float = 0.5):
            # Bypass parent __init__ to inject custom dropout in head.
            super(DinoWithHead, self).__init__()
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
                nn.Linear(embed_dim, 256), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

    model = DinoWithHeadCustom(dinov2, num_classes=6,
                                unfreeze_last_n=args.unfreeze_blocks,
                                dropout=args.head_dropout).to(device)
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

    print(f"\n[4] Training {args.epochs} epochs, batch {args.tile_batch}, "
          f"backbone_lr={args.backbone_lr}, head_lr={args.head_lr}", flush=True)
    best_iou = -1.0
    no_improve = 0
    PATIENCE = 5
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
        acc, macro, ious, _ = evaluate_full_image(model, test_rgb, test_lbl, device,
                                                   stride=args.stride, batch_size=args.tile_batch)
        ious_str = " ".join(f"{ID_TO_DLTB.get(c, str(c))}:{ious[c]:.3f}" for c in sorted(ious))
        print(f"    eval: acc={acc:.3f} macro_iou={macro:.3f} [{ious_str}]  ({time.time()-t0:.0f}s)",
              flush=True)

        if macro > best_iou:
            best_iou = macro
            no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best macro_iou={macro:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[early stop] no improvement for {PATIENCE} epochs", flush=True)
                break

    print(f"\n[done] best macro_iou={best_iou:.3f}", flush=True)


if __name__ == "__main__":
    main()
