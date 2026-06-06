"""Control experiment: does 1m Esri fusion beat 10m-only on the SAME cells/grid?

Trains an smp U-Net on the fused POC cells (5m, 12-ch) and reports cropland F1 on
the held-out test split. Run it TWICE on identical data:
  --channels 12  -> Esri 1m RGB (3) + 10m S2 RGBNIR/NDVI/4yr (9)
  --channels  9  -> drop the 3 hires bands (10m-only baseline at the same 5m grid)
The F1 delta isolates the 1m contribution (data quantity / grid held constant).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")


class FusedDS(torch.utils.data.Dataset):
    def __init__(self, names, d, channels, training):
        self.names = names; self.d = Path(d); self.ch = channels; self.training = training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        z = np.load(self.d / f"{self.names[i]}.npz")
        x = z["x12"].astype(np.float32)
        lbl = z["label"].astype(np.int64)
        if self.ch == 9:
            x = x[3:]                      # drop the 3 Esri hires bands
        if self.training:
            if np.random.rand() < 0.5: x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.rand() < 0.5: x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k: x = np.rot90(x, k, axes=(1, 2)).copy(); lbl = np.rot90(lbl, k).copy()
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lbl))


def f1_crop(model, loader, device):
    model = getattr(model, "eval")(); tp = fp = fn = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            p = model(xb).argmax(1); v = yb > 0
            pi = (p == 1) & v; ti = (yb == 1) & v
            tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9), prec, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fused-dir", default=str(HOME / "data/fused_poc"))
    p.add_argument("--channels", type=int, default=12, choices=[9, 12])
    p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

    man = json.loads((Path(a.fused_dir) / "manifest.json").read_text())
    tr = FusedDS(man["train"], a.fused_dir, a.channels, True)
    te = FusedDS(man["test"], a.fused_dir, a.channels, False)
    print(f"[fusion] channels={a.channels} backbone={a.backbone} train={len(tr)} test={len(te)}", flush=True)
    trl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    tel = torch.utils.data.DataLoader(te, batch_size=a.batch_size, shuffle=False, num_workers=2)

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=a.backbone, encoder_weights="imagenet",
                     in_channels=a.channels, classes=3).to(a.device)

    bc = np.zeros(3)
    for n in man["train"][:300]:
        bc += np.bincount(np.load(Path(a.fused_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1.0 / np.sqrt(bc), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for xb, yb in trl:
            xb = xb.to(a.device); yb = yb.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(xb)
                loss = F.cross_entropy(lg.float(), yb, weight=cwt, ignore_index=0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        f1, pr, rc = f1_crop(model, tel, a.device)
        best = max(best, f1)
        print(f"  ep{ep+1}/{a.epochs} F1={f1:.4f} P={pr:.3f} R={rc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] channels={a.channels} seed={a.seed} best_test_F1={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
