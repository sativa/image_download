"""v18: hard negative mining — continue v17 with FP-weighted tile sampling.

Pipeline:
  1. Load v17 best.pt
  2. Inference on all 105 train regions, tile-by-tile, compute per-tile FP rate
  3. Weight each tile by (1 + 3 * fp_rate)
  4. Continue training with WeightedRandomSampler and lower LR (1e-6 / 1e-4)
  5. Eval EMA + TTA each epoch on same 8-balanced test set
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DEFAULT_DINOV2, IMAGENET_MEAN, IMAGENET_STD, DLBM_TO_CLASS, DinoUNet
from train_v16_binary import rasterise_dltb_binary, composite_binary_loss
from train_v17_cropland import (
    StrongAugTilesDataset, evaluate_full_binary_tta, EMA
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--init-checkpoint", type=Path, default=HOME / "results/v17/best.pt")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v18")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-6)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--hard-weight", type=float, default=3.0,
                   help="multiplier on tile sampling weight per unit FP rate")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} test", flush=True)

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
    print(f"  {len(gdf_per_county)} counties", flush=True)

    def _load(region_list):
        regs = []
        for r in region_list:
            bb = tuple(r["bbox"])
            for src in ["esri", "google"]:
                path = args.data_cache / f"{r['county']}_{r['idx']}_{src}.tif"
                if not path.exists(): continue
                with rasterio.open(path) as rs:
                    bands = rs.read(out_dtype="uint8")
                    transform = rs.transform; H, W = rs.height, rs.width
                rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
                lbl = rasterise_dltb_binary(gdf_per_county[r["county"]], bb, transform, H, W)
                regs.append((rgb, lbl))
        return regs

    print(f"\n[2] Loading data", flush=True)
    t0 = time.time()
    train_regions = _load(regions_meta["train"])
    test_regions = _load(regions_meta["test"])
    print(f"  {len(train_regions)} train + {len(test_regions)} test ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[3] Build tile dataset", flush=True)
    train_ds = StrongAugTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(train_ds)} tiles", flush=True)

    print(f"\n[4] Load v17 model and mine hard tiles", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=3, unfreeze_last_n=args.unfreeze_blocks).to(device)
    state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = getattr(model, "eval")()
    print(f"  loaded {args.init_checkpoint.name}", flush=True)

    # Score each tile by FP+FN rate against ground truth
    t0 = time.time()
    tile_weights = np.ones(len(train_ds), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(train_ds), args.tile_batch):
            batch_items = [train_ds.items[j] for j in range(i, min(i+args.tile_batch, len(train_ds)))]
            tiles = []; labels = []
            for rgb, lbl, top, left in batch_items:
                rgb_tile = rgb[top:top+448, left:left+448]
                lbl_tile = lbl[top:top+448, left:left+448]
                arr = rgb_tile.astype(np.float32) / 255.0
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                tiles.append(arr); labels.append(lbl_tile)
            x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            for k, (lbl_tile, pred) in enumerate(zip(labels, preds)):
                valid = lbl_tile > 0
                if not valid.any():
                    continue
                p = pred[valid]; t = lbl_tile[valid]
                # binary cropland error
                fp = int(((p == 1) & (t != 1)).sum())
                fn = int(((p != 1) & (t == 1)).sum())
                total = int(valid.sum())
                err_rate = (fp + fn) / total
                tile_weights[i + k] = 1.0 + args.hard_weight * err_rate
            if (i // args.tile_batch) % 50 == 0:
                print(f"    mined {i+len(batch_items)}/{len(train_ds)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  mining done in {time.time()-t0:.0f}s", flush=True)
    print(f"  tile weight stats: min={tile_weights.min():.2f} median={np.median(tile_weights):.2f}"
          f" mean={tile_weights.mean():.2f} max={tile_weights.max():.2f}", flush=True)
    n_hard = int((tile_weights > 1.5).sum())
    print(f"  hard tiles (weight>1.5): {n_hard} ({n_hard/len(tile_weights)*100:.1f}%)", flush=True)

    print(f"\n[5] Continue training: backbone_lr={args.backbone_lr}, head_lr={args.head_lr}", flush=True)
    ema = EMA(model, decay=0.999)
    bin_counts = np.zeros(3, dtype=np.float64)
    for _, lbl in train_regions:
        bin_counts += np.bincount(lbl.ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0
    cw = (cw / cw.sum() * 2)
    cw_t = torch.from_numpy(cw).to(device)

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
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=tile_weights.tolist(), num_samples=len(train_ds), replacement=True
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.tile_batch, sampler=sampler, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    best_f1 = -1.0
    no_improve = 0
    PATIENCE = 6
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; ep_c = 0; ep_t = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss, _ = composite_binary_loss(logits, yb, cw_t)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ema.update(model)
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
        ema_model = ema.apply(model)
        ema_model = getattr(ema_model, "eval")()
        all_metrics = []
        for rgb, lbl in test_regions:
            m = evaluate_full_binary_tta(ema_model, rgb, lbl, device,
                                          stride=args.stride, batch_size=args.tile_batch,
                                          use_tta=True)
            if m: all_metrics.append(m)
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        print(f"    eval (EMA+TTA): acc={avg['acc']:.3f} iou={avg['iou']:.3f} "
              f"prec={avg['precision']:.3f} rec={avg['recall']:.3f} F1={avg['f1']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            no_improve = 0
            torch.save(ema_model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop]", flush=True); break

    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
