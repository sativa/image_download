"""v39: DINOv2-L + 9-ch multi-temporal (4 yr NDVI + S2 RGBNIR + S2 NDVI).

Tests whether DINOv2-L finally beats EfNet-B5 when given multi-temporal signal.
Previous attempts: v24/v24+ used only S2 5-ch and lost to B3 by ~0.07-0.02.

Architecture: DinoUNet5ch with in_channels=9 (extends DINOv2 patch_embed 3→9).
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
from train_v12_unet import DLBM_TO_CLASS, DEFAULT_DINOV2
from train_v24_dino_s2 import DinoUNet5ch
from train_v33_multitemporal import (
    S2MultiTempDataset, EXTRA_YEARS, S2_MEAN, S2_STD,
    NDVI_MEAN, NDVI_STD, upsample_to
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--ndvi-yr-dir", type=Path, default=HOME / "data/v33_ndvi_multitemporal")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v39")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    regions = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions['train'])} train + {len(regions['test'])} test", flush=True)

    # PARALLEL loader (use fast_load_multitemp)
    from fast_load_multitemp import parallel_loadsplit_multitemp
    t0 = time.time()
    train_cells, sk_t = parallel_loadsplit_multitemp(
        regions["train"], args.dltb_cache, args.s2_dir, args.ndvi_yr_dir,
        extra_years=EXTRA_YEARS, max_workers=16,
    )
    test_cells, sk_v = parallel_loadsplit_multitemp(
        regions["test"], args.dltb_cache, args.s2_dir, args.ndvi_yr_dir,
        extra_years=EXTRA_YEARS, max_workers=8,
    )
    print(f"  train={len(train_cells)} test={len(test_cells)} load={time.time()-t0:.0f}s",
          flush=True)

    train_ds = S2MultiTempDataset(train_cells, args.target_size, training=True)
    test_ds = S2MultiTempDataset(test_cells, args.target_size, training=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)

    print(f"\n[2] DINOv2-L + UNet decoder, 9-ch", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=9,
                        unfreeze_last_n=args.unfreeze_blocks).to(args.device)
    print(f"  total {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M",
          flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0/np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    cw_t = torch.from_numpy(cw).to(args.device)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("backbone")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone")]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(args.device); yb = yb.to(args.device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear",
                                            align_corners=False)
                loss = F.cross_entropy(logits.float(), yb, weight=cw_t, ignore_index=0)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n_b += 1
        sched.step()

        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(args.device); yb = yb.to(args.device)
                logits = model(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear",
                                            align_corners=False)
                p_ = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    pi = (p_==1) & v; ti = (yb==1) & v
                    tp += int((pi & ti).sum().item()); fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item()); tn += int((~pi & ~ti & v).sum().item())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0
        acc = (tp+tn)/max(tp+fp+fn+tn,1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} | acc={acc:.3f} "
              f"iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 8: print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
