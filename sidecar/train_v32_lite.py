"""v32_lite: v31 base + boundary auxiliary loss + longer training.

Additions over v31:
  - Boundary aux loss: 3×3 morphological gradient of GT mask vs predicted mask
    → forces sharper field edges, important for cropland precision
  - More epochs (40 vs 20) + larger patience (12 vs 8)
  - Optional lower LR for stability with extra loss term
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
from train_v12_unet import DLBM_TO_CLASS
from train_v31_mask2former import (
    S2CellDataset, patch_swin_5ch, make_targets, m2f_to_semantic, evaluate
)


def boundary_from_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological gradient = dilation - erosion.
    Input: mask of shape (B, H, W) float ∈ [0, 1].
    Returns: boundary mask (B, H, W) where 1 = edge pixels.
    """
    m = mask.unsqueeze(1)  # B, 1, H, W
    pad = kernel_size // 2
    dil = F.max_pool2d(m, kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-m, kernel_size, stride=1, padding=pad)
    return (dil - ero).squeeze(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v32_lite")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--boundary-weight", type=float, default=0.3,
                   help="weight of boundary aux loss")
    p.add_argument("--m2f-model", default="facebook/mask2former-swin-base-ade-semantic")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"[1] load Mask2Former {args.m2f_model}", flush=True)
    from transformers import Mask2FormerForUniversalSegmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.m2f_model,
        num_labels=3,
        id2label={0: "nodata", 1: "crop", 2: "other"},
        label2id={"nodata": 0, "crop": 1, "other": 2},
        ignore_mismatched_sizes=True,
    )
    patch_swin_5ch(model, in_channels=5)
    model = model.to(device)
    print(f"  total params: {sum(pp.numel() for pp in model.parameters())/1e6:.1f}M", flush=True)

    print(f"[2] parallel load cells", flush=True)
    from fast_load_s2 import parallel_loadsplit
    regions = json.loads(args.regions_json.read_text())
    train_cells, sk_t = parallel_loadsplit(regions["train"], args.dltb_cache,
                                             args.s2_dir, max_workers=16)
    test_cells, sk_v = parallel_loadsplit(regions["test"], args.dltb_cache,
                                            args.s2_dir, max_workers=8)
    print(f"  train: {len(train_cells)} | test: {len(test_cells)}", flush=True)

    train_ds = S2CellDataset(train_cells, args.target_size, training=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0; PATIENCE = 12; B_WT = args.boundary_weight

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; ep_bdy = 0; n_b = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            mask_labels, class_labels = [], []
            for b in range(yb.shape[0]):
                ml, cl = make_targets(yb[b], num_classes=3, ignore_index=0)
                mask_labels.append(ml); class_labels.append(cl)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(pixel_values=xb,
                                 mask_labels=mask_labels,
                                 class_labels=class_labels)
                main_loss = outputs.loss
                sem = m2f_to_semantic(outputs.class_queries_logits,
                                       outputs.masks_queries_logits,
                                       xb.shape[-2:])
                crop_prob = sem[:, 1]
                pred_bdy = boundary_from_mask(crop_prob, kernel_size=3)
                gt_crop = (yb == 1).float()
                gt_bdy = boundary_from_mask(gt_crop, kernel_size=3)
                valid = (yb > 0).float()
            # Boundary loss in fp32 OUTSIDE autocast (BCE+log unsafe in fp16)
            p32 = pred_bdy.float().clamp(1e-4, 1 - 1e-4)
            g32 = gt_bdy.float()
            v32 = valid.float()
            bdy_loss = -(g32 * torch.log(p32) + (1 - g32) * torch.log(1 - p32))
            bdy_loss = (bdy_loss * v32).sum() / v32.sum().clamp(min=1)
            loss = main_loss.float() + B_WT * bdy_loss
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim); scaler.update()
            ep_loss += main_loss.item(); ep_bdy += bdy_loss.item(); n_b += 1
        sched.step()

        m = evaluate(model, test_cells, device, target_size=args.target_size,
                     batch_size=args.batch_size)
        print(f"  ep{ep+1}/{args.epochs}: m2f={ep_loss/max(n_b,1):.4f} bdy={ep_bdy/max(n_b,1):.4f} "
              f"| acc={m['acc']:.3f} iou={m['iou']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} F1={m['f1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE: print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
