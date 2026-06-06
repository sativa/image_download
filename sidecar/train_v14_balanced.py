"""v14: same as v13 but with BALANCED test set (all 5 classes ≥3% each).

Test set: 8 cells across 8 unseen counties, each balanced. This gives
a fair cross-county macro-IoU number (v11-v13 were biased by test
sets with 0% 园地, dragging IoU artificially low).

Everything else identical to v13: DINOv2-large + UNet decoder + fp16
+ composite CE+Lovász+Dice loss + class-weighted, last 4 blocks
unfrozen.
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
    DEFAULT_DINOV2, DLTB_CLASS_TO_ID, ID_TO_DLTB, IMAGENET_MEAN, IMAGENET_STD,
    DLBM_TO_CLASS, rasterise_dltb_region, PixelTilesDataset, DinoUNet,
    evaluate_full_image,
)
from train_v13_lovasz import composite_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v11_regions_balanced.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v14")
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} BALANCED test", flush=True)

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
                lbl = rasterise_dltb_region(gdf_per_county[r["county"]], bb, transform, H, W)
                regs.append((rgb, lbl))
        return regs

    print("\n[2] Loading + rasterising", flush=True)
    t0 = time.time()
    train_regions = _load(regions_meta["train"])
    test_regions = _load(regions_meta["test"])
    print(f"  loaded {len(train_regions)} train + {len(test_regions)} test ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[3] Pixel tile dataset", flush=True)
    train_ds = PixelTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(train_ds)} tiles", flush=True)
    bin_counts = np.zeros(6, dtype=np.float64)
    for _, lbl in train_regions:
        bin_counts += np.bincount(lbl.ravel(), minlength=6)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0)
    cw[0] = 0.0
    cw = (cw / cw.sum() * 5).astype(np.float32)
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)

    print(f"\n[4] Building model", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=6, unfreeze_last_n=args.unfreeze_blocks).to(device)
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

    print(f"\n[5] Training {args.epochs} epochs, batch {args.tile_batch}, fp16 + composite loss", flush=True)
    best_iou = -1.0
    no_improve = 0
    PATIENCE = 5
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_c = 0; ep_t = 0; n_b = 0
        ep_breakdown = {"ce": 0, "lov": 0, "dice": 0}
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss, parts = composite_loss(logits, yb, cw_t, num_classes=6)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ep_loss += loss.item(); n_b += 1
            for k in ep_breakdown:
                ep_breakdown[k] += parts[k]
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                valid = yb > 0
                if valid.any():
                    ep_c += int((preds[valid] == yb[valid]).sum().item())
                    ep_t += int(valid.sum().item())
        sched.step()
        train_acc = ep_c / max(ep_t, 1)
        bd = " ".join(f"{k}={v/n_b:.3f}" for k, v in ep_breakdown.items())
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} ({bd}) train_acc={train_acc:.3f} ({time.time()-t0:.0f}s)",
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
        print(f"    eval: acc={avg_acc:.3f} iou={avg_macro:.3f} [{per_cls_str}] ({time.time()-t0:.0f}s)",
              flush=True)
        if avg_macro > best_iou:
            best_iou = avg_macro
            no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best={avg_macro:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[early stop]", flush=True)
                break

    print(f"\n[done] best={best_iou:.3f}", flush=True)


if __name__ == "__main__":
    main()
