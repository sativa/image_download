"""Route (c) Stage-1: 1m boundary segmentation from multi-source (Esri+Google, 6ch).

smp U-Net on 512px@1m crops -> cropland extent, with a boundary loss for sharp edges.
Trains on c_1m/*.npz (from fuse_1m.py). Eval = 1m cropland F1 (diagnostic; the headline
number comes from stage-2 which adds 10m spectral). best.pt feeds stage-2.
"""
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")


# channel slices of the fused x6 (esri RGB = 0:3, google RGB = 3:6)
SRC_CH = {"both": [0, 1, 2, 3, 4, 5], "esri": [0, 1, 2], "google": [3, 4, 5]}


class S1DS(torch.utils.data.Dataset):
    def __init__(self, names, d, crop, training, channels):
        self.n = names; self.d = Path(d); self.c = crop; self.tr = training; self.ch = channels

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        z = np.load(self.d / f"{self.n[i]}.npz")
        x = z["x6"][self.ch].astype(np.float32) / 255.0
        l = z["label"].astype(np.int64)
        C, H, W = x.shape; cs = self.c
        if H < cs or W < cs:
            ph = max(0, cs - H); pw = max(0, cs - W)
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
            l = np.pad(l, ((0, ph), (0, pw)), mode="constant")
            H, W = x.shape[1], x.shape[2]
        if self.tr:
            t = random.randint(0, H - cs); le = random.randint(0, W - cs)
        else:
            t = (H - cs) // 2; le = (W - cs) // 2
        x = x[:, t:t + cs, le:le + cs]; l = l[t:t + cs, le:le + cs]
        if self.tr:
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); l = l[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1, :].copy(); l = l[::-1, :].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2)).copy(); l = np.rot90(l, k).copy()
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(l))


def boundary(m, k=3):
    m = m.unsqueeze(1).float(); p = k // 2
    d = F.max_pool2d(m, k, 1, p); e = -F.max_pool2d(-m, k, 1, p)
    return (d - e).squeeze(1)


def f1_crop(model, loader, dev):
    model = getattr(model, "eval")(); tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); y = y.to(dev); p = model(x).argmax(1); v = y > 0
            pi = (p == 1) & v; ti = (y == 1) & v
            tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(HOME / "data/c_1m"))
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bdy-w", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(HOME / "results/c_stage1"))
    p.add_argument("--sources", default="both", choices=["both", "esri", "google"],
                   help="which 1m source(s): both=Esri+Google 6ch, esri/google=3ch (dual-vs-single ablation)")
    a = p.parse_args()
    CH = SRC_CH[a.sources]
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    trl = torch.utils.data.DataLoader(S1DS(man["train"], a.data_dir, a.crop, True, CH),
                                      batch_size=a.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    tel = torch.utils.data.DataLoader(S1DS(man["test"], a.data_dir, a.crop, False, CH),
                                      batch_size=a.batch_size, shuffle=False, num_workers=2)
    print(f"[c-stage1] sources={a.sources} ({len(CH)}ch)@1m train={len(man['train'])} test={len(man['test'])} crop={a.crop}", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=a.backbone, encoder_weights="imagenet", in_channels=len(CH), classes=3).to(a.device)
    bc = np.zeros(3)
    for n in man["train"][:200]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1.0 / np.sqrt(bc), 0.0).astype(np.float32); cw[0] = 0.0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for x, y in trl:
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(x)
                ce = F.cross_entropy(lg.float(), y, weight=cwt, ignore_index=0)
                cp = torch.softmax(lg.float(), 1)[:, 1]
                pb = boundary(cp).clamp(1e-4, 1 - 1e-4); gb = boundary((y == 1).float()); vv = (y > 0).float()
                bl = -(gb * torch.log(pb) + (1 - gb) * torch.log(1 - pb))
                bl = (bl * vv).sum() / vv.sum().clamp(min=1)
                loss = ce + a.bdy_w * bl
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        f1, pr, rc = f1_crop(model, tel, a.device)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), Path(a.out) / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} 1m-F1={f1:.4f} P={pr:.3f} R={rc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] c-stage1 best 1m-F1={best:.4f} -> {a.out}/best.pt", flush=True)


if __name__ == "__main__":
    main()
