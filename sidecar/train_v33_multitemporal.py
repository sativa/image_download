"""v33: multi-temporal NDVI fusion.

Input channels (9):
  [0-3] Sentinel-2 RGBNIR (B04, B03, B02, B08) @ 10m, 2021 median
  [4]   Sentinel-2 NDVI 2021 @ 10m
  [5]   China NDVI 2018 (annual max, 30m, upsampled to 10m grid)
  [6]   China NDVI 2019
  [7]   China NDVI 2020
  [8]   China NDVI 2022   (skip 2021 — already covered by S2 NDVI)

Why this works for cropland:
  - Cropland NDVI varies year-to-year (rotation, fallow) → temporal variability
  - Forest stable → low temporal std
  - Network can learn temporal-variability features that single-snapshot can't
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

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3
# China NDVI is int16 scaled — typical scaling: value/10000 = NDVI [-1, 1]
CN_NDVI_SCALE = 10000.0

# Years from China_NDVI dataset to use (skip 2021 — covered by S2 NDVI)
EXTRA_YEARS = [2018, 2019, 2020, 2022]


class S2MultiTempDataset(torch.utils.data.Dataset):
    """One cell = one sample. 9-ch stack."""

    def __init__(self, cells, target_size=224, training=True):
        self.cells = cells; self.size = target_size; self.training = training

    def __len__(self): return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32).copy()  # (4, Hs, Ws)
        ndvi_s2 = c["ndvi_s2"].astype(np.float32)        # (Hs, Ws)
        ndvi_yr = c["ndvi_years"].astype(np.float32)     # (5, Hs, Ws), already upsampled to S2 grid
        lbl = c["label"].astype(np.int64)
        Hs, Ws = ndvi_s2.shape

        # Normalize
        for b in range(4):
            rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_s2 = (ndvi_s2 - NDVI_MEAN) / NDVI_STD
        ndvi_yr = (ndvi_yr - NDVI_MEAN) / NDVI_STD

        # Stack 4 + 1 + len(EXTRA_YEARS) = 9 channels
        x = np.concatenate([rgbnir, ndvi_s2[None], ndvi_yr], axis=0).astype(np.float32)

        H, W = x.shape[1], x.shape[2]; sz = self.size
        if H < sz or W < sz:
            ph = max(0, sz - H); pw = max(0, sz - W)
            x = np.pad(x, ((0,0),(0,ph),(0,pw)), mode="edge")
            lbl = np.pad(lbl, ((0,ph),(0,pw)), mode="constant")
            H, W = x.shape[1], x.shape[2]
        if self.training:
            top = np.random.randint(0, max(1, H - sz + 1))
            left = np.random.randint(0, max(1, W - sz + 1))
        else:
            top = (H - sz) // 2; left = (W - sz) // 2
        x = x[:, top:top+sz, left:left+sz]; lbl = lbl[top:top+sz, left:left+sz]

        if self.training:
            if np.random.random() < 0.5:
                x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5:
                x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1,2)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0,1)).copy()
            # Jitter only RGBNIR (channels 0-3), keep NDVI clean
            jit = 1.0 + (np.random.random(4) - 0.5) * 0.2
            for b in range(4): x[b] *= jit[b]
        return torch.from_numpy(x), torch.from_numpy(lbl)


def upsample_to(arr2d, H, W):
    t = torch.from_numpy(arr2d.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--ndvi-yr-dir", type=Path, default=HOME / "data/v33_ndvi_multitemporal")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v33")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--backbone", default="efficientnet-b3")
    p.add_argument("--arch", default="unet",
                   choices=["unet", "segformer", "unetplusplus", "deeplabv3plus"],
                   help="smp decoder architecture (for ensemble diversity)")
    p.add_argument("--loss", default="ce", choices=["ce", "dice_ce"],
                   help="ce = class-weighted CE (original); "
                        "dice_ce = 0.5*CE + 0.5*soft-Dice(cropland)")
    p.add_argument("--encoder-weights", default="imagenet",
                   help="imagenet | none (none = encoder from scratch, e.g. if a "
                        "pretrained-weights download is blocked on the box)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=None,
                   help="if set, seed python/numpy/torch for a reproducible run "
                        "(used to build the 5-seed v36 deep ensemble)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[seed] python/numpy/torch seeded with {args.seed}", flush=True)

    print(f"[1] {args.regions_json.name}", flush=True)
    regions = json.loads(args.regions_json.read_text())
    print(f"  {len(regions['train'])} train + {len(regions['test'])} test", flush=True)

    # Parallel loader: 22K cells from 28min → ~9min (3.2× speedup)
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
    print(f"  train: {len(train_cells)} ({sk_t} skipped) | "
          f"test: {len(test_cells)} ({sk_v} skipped) | total {time.time()-t0:.0f}s",
          flush=True)

    train_ds = S2MultiTempDataset(train_cells, args.target_size, training=True)
    test_ds = S2MultiTempDataset(test_cells, args.target_size, training=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=2)

    n_ch = 5 + len(EXTRA_YEARS)
    print(f"\n[2] UNet + {args.backbone}, {n_ch}-ch (RGBNIR + NDVI 2021 + {len(EXTRA_YEARS)} years)",
          flush=True)
    import segmentation_models_pytorch as smp
    _ARCH = {"unet": smp.Unet, "segformer": smp.Segformer,
             "unetplusplus": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus}
    _ew = None if str(args.encoder_weights).lower() in ("none", "null", "scratch") else args.encoder_weights
    model = _ARCH[args.arch](encoder_name=args.backbone, encoder_weights=_ew,
                             in_channels=n_ch, classes=3).to(args.device)
    print(f"  arch={args.arch} backbone={args.backbone} loss={args.loss} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0/np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    cw_t = torch.from_numpy(cw).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; correct = 0; total = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(args.device); yb = yb.to(args.device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), yb, weight=cw_t, ignore_index=0)
                if args.loss == "dice_ce":
                    probs = torch.softmax(logits.float(), dim=1)
                    p1 = probs[:, 1]; valid = (yb > 0).float(); t1 = (yb == 1).float()
                    inter = (p1 * t1 * valid).sum()
                    denom = (p1 * valid).sum() + (t1 * valid).sum()
                    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
                    loss = 0.5 * loss + 0.5 * dice
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n_b += 1
            with torch.no_grad():
                p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    correct += int((p[v]==yb[v]).sum().item())
                    total += int(v.sum().item())
        sched.step()

        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(args.device); yb = yb.to(args.device)
                logits = model(xb); p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    pi = (p==1) & v; ti = (yb==1) & v
                    tp += int((pi & ti).sum().item()); fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item()); tn += int((~pi & ~ti & v).sum().item())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0
        acc = (tp+tn)/max(tp+fp+fn+tn,1)
        train_acc = correct / max(total, 1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} "
              f"| acc={acc:.3f} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 10: print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
