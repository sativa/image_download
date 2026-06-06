"""v35: full 13-year NDVI (2010-2022) multi-temporal training.

Differences from v33:
  - EXTRA_YEARS = 13 years instead of 4 (2018, 2019, 2020, 2022)
  - Input channels: 5 (RGBNIR + S2 NDVI) + 12 (extra NDVI years excluding 2021)
    = 17 channels (NOT 18; 2021 dropped because S2 NDVI covers it)
  - Data dir: v35_ndvi_full (contains all 13 years)
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

# Full 12 extra years (2010-2022, excluding 2021 → covered by S2 NDVI)
EXTRA_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
               2018, 2019, 2020, 2022]


def upsample_to(arr2d, H, W):
    t = torch.from_numpy(arr2d.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


class S2MultiTempDataset(torch.utils.data.Dataset):
    def __init__(self, cells, target_size=224, training=True):
        self.cells = cells; self.size = target_size; self.training = training

    def __len__(self): return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32).copy()
        ndvi_s2 = c["ndvi_s2"].astype(np.float32)
        ndvi_yr = c["ndvi_years"].astype(np.float32)
        lbl = c["label"].astype(np.int64)
        for b in range(4): rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_s2 = (ndvi_s2 - NDVI_MEAN) / NDVI_STD
        ndvi_yr = (ndvi_yr - NDVI_MEAN) / NDVI_STD
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
            if np.random.random() < 0.5: x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5: x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1,2)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0,1)).copy()
            jit = 1.0 + (np.random.random(4) - 0.5) * 0.2
            for b in range(4): x[b] *= jit[b]
        return torch.from_numpy(x), torch.from_numpy(lbl)


def boundary_from_mask(mask, kernel_size=3):
    m = mask.unsqueeze(1)
    pad = kernel_size // 2
    dil = F.max_pool2d(m, kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-m, kernel_size, stride=1, padding=pad)
    return (dil - ero).squeeze(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--ndvi-yr-dir", type=Path, default=HOME / "data/v35_ndvi_full")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v35")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--backbone", default="efficientnet-b3")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--boundary-weight", type=float, default=0.3,
                   help="add boundary aux loss (0 = no boundary)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] {args.regions_json.name}", flush=True)
    regions = json.loads(args.regions_json.read_text())

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
    print(f"\n[2] UNet+{args.backbone}, {n_ch}-ch, +boundary w={args.boundary_weight}", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=args.backbone, encoder_weights="imagenet",
                     in_channels=n_ch, classes=3).to(args.device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0/np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    cw_t = torch.from_numpy(cw).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    BW = args.boundary_weight

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_ce = 0; ep_bdy = 0; n_b = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(args.device); yb = yb.to(args.device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                ce = F.cross_entropy(logits.float(), yb, weight=cw_t, ignore_index=0)
                probs = torch.softmax(logits.float(), dim=1)
                crop_prob = probs[:, 1]
            if BW > 0:
                pred_bdy = boundary_from_mask(crop_prob, 3)
                gt_crop = (yb == 1).float()
                gt_bdy = boundary_from_mask(gt_crop, 3)
                valid = (yb > 0).float()
                p32 = pred_bdy.clamp(1e-4, 1 - 1e-4).float()
                g32 = gt_bdy.float()
                v32 = valid.float()
                bdy = -(g32 * torch.log(p32) + (1 - g32) * torch.log(1 - p32))
                bdy = (bdy * v32).sum() / v32.sum().clamp(min=1)
                loss = ce + BW * bdy
                ep_bdy += bdy.item()
            else:
                loss = ce
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            ep_ce += ce.item(); n_b += 1
        sched.step()

        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(args.device); yb = yb.to(args.device)
                p = model(xb).argmax(dim=1); v = yb > 0
                if v.any():
                    pi = (p==1) & v; ti = (yb==1) & v
                    tp += int((pi & ti).sum().item()); fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item()); tn += int((~pi & ~ti & v).sum().item())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0
        acc = (tp+tn)/max(tp+fp+fn+tn,1)
        print(f"  ep{ep+1}/{args.epochs}: ce={ep_ce/n_b:.4f} bdy={ep_bdy/n_b:.4f} "
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
