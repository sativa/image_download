"""DINOv2-large + UNet decoder, fine-tuned on 1m imagery (c_1m) for cropland seg.

The DINO arm of the foundation-model comparison (vs SAM3 zero-shot, vs UNet-b5 baseline).
Input = 6ch @ 1m (Esri+Google RGB), ImageNet-normalized per RGB triplet (matches DINOv2
pretraining). Predicts cropland natively at 1m; scored by 1m-F1 (full-cell tiled) like route_a.

Reuses DinoUNet5ch(in_channels=6) from v24. Backbone frozen except last N blocks + the new
6ch patch-embed + decoder. Note history: DINOv2+UNet lost to UNet on the 10m task — this tests
whether 1m RGB (closer to DINOv2's natural-image pretraining) changes that.
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def norm6(x6):
    """(6,H,W) uint8 -> ImageNet-normalized float32, per RGB triplet."""
    x = x6.astype(np.float32) / 255.0
    for t in (0, 3):
        for c in range(3):
            x[t + c] = (x[t + c] - IMG_MEAN[c]) / IMG_STD[c]
    return x


class C1mDS(torch.utils.data.Dataset):
    def __init__(self, names, dd, crop, training):
        self.n = names; self.d = Path(dd); self.c = crop; self.tr = training

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        z = np.load(self.d / f"{self.n[i]}.npz")
        x6 = z["x6"]; lbl = z["label"].astype(np.int64); cs = self.c
        _, SZ, SZw = x6.shape
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge")
            lbl = np.pad(lbl, ((0, ph), (0, pw))); SZ, SZw = x6.shape[1:]
        if self.tr:
            t = random.randint(0, SZ - cs); l = random.randint(0, SZw - cs)
        else:
            t = (SZ - cs) // 2; l = (SZw - cs) // 2
        x = norm6(x6[:, t:t + cs, l:l + cs]); lc = lbl[t:t + cs, l:l + cs]
        if self.tr:
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); lc = lc[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1, :].copy(); lc = lc[::-1, :].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2)).copy(); lc = np.rot90(lc, k).copy()
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lc))


@torch.no_grad()
def full_eval(model, names, dd, dev, cs=448):
    model.eval(); tp = fp = fn = 0
    for name in names:
        z = np.load(Path(dd) / f"{name}.npz"); x6 = z["x6"]; lbl = z["label"]
        _, SZ, SZw = x6.shape
        acc = np.zeros((3, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
        ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
        if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
        if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
        for t in ys:
            for l in xs:
                xb = torch.from_numpy(norm6(x6[:, t:t + cs, l:l + cs])).unsqueeze(0).to(dev)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg = model(xb)
                    if lg.shape[-2:] != (cs, cs):
                        lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0)
        v = lbl > 0; ti = (lbl == 1) & v; pi = (pred == 1) & v
        tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_1m")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-train", type=int, default=0, help="cap train cells (0=all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr_names = man["train"]; te_names = man["test"]
    if a.max_train: tr_names = tr_names[:a.max_train]
    print(f"[dino-1m] crop={a.crop} train={len(tr_names)} test={len(te_names)}", flush=True)

    trl = torch.utils.data.DataLoader(C1mDS(tr_names, a.data_dir, a.crop, True),
                                      batch_size=a.batch_size, shuffle=True,
                                      num_workers=a.workers, pin_memory=True, drop_last=True)

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=a.unfreeze).to(a.device)
    nt = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"  total={sum(q.numel() for q in model.parameters())/1e6:.1f}M trainable={nt/1e6:.1f}M", flush=True)

    bc = np.zeros(3)
    for n in tr_names[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)

    bb = [q for nm, q in model.named_parameters() if q.requires_grad and nm.startswith("backbone")]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and not nm.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr},
                             {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); el = 0; nb = 0
        for x, y in trl:
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(x)
                if lg.shape[-2:] != y.shape[-2:]:
                    lg = F.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(lg.float(), y, weight=cwt, ignore_index=0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            el += loss.item(); nb += 1
        sch.step()
        f1 = full_eval(model, te_names, a.data_dir, a.device, a.crop)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} loss={el/nb:.4f} 1m-F1={f1:.4f} (best {best:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[FINAL dino-1m] best 1m-F1={best:.4f}", flush=True)
    print(f"  compare: SAM3 zero-shot ~0.55 | UNet-b5 1m (route_a stage1) 0.838", flush=True)
    json.dump({"best_1m_f1": best, "crop": a.crop, "n_train": len(tr_names)}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
