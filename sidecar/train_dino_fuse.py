"""DINOv2 + 15ch (1m RGB + 10m spectral) — the user's intended design: 10m spectral as
"icing on the cake" ON TOP of the 1m DINOv2 fine-tune.

vs train_dino_1m.py (6ch, 1m-only, 1m-F1 0.860 in-domain / 0.843 cross-province). Tests whether
fusing the 9-ch 10m spectral (upsampled) lifts in-domain F1 past 0.86. CAVEAT (to verify): the
1m-RGB-only model generalizes cross-province because RGB texture is domain-invariant; the 10m
spectral carries Gansu-specific signal (pure-10m collapses to 0.236 cross-province), so fusion may
trade in-domain gain for cross-province loss. Eval cross-province separately (dino_fuse_changzhi).

DINOv2 patch_embed extended to 15ch; the 6 1m-RGB channels get ImageNet norm (match DINOv2
pretraining), the 9 spectral channels keep route_a's S2/NDVI norm (already in build_spec).
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_route_a import build_spec, assemble15, EXTRA_YEARS
from fast_load_multitemp import parallel_loadsplit_multitemp
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def imnorm15(x15):
    """In-place ImageNet-norm the two 1m-RGB triplets (channels 0:3, 3:6); spectral 6:15 unchanged."""
    for t in (0, 3):
        for c in range(3):
            x15[t + c] = (x15[t + c] - IMG_MEAN[c]) / IMG_STD[c]
    return x15


class FuseDS(torch.utils.data.Dataset):
    def __init__(self, names, dd, spec, crop, training):
        self.n = names; self.d = Path(dd); self.spec = spec; self.c = crop; self.tr = training

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        name = self.n[i]; z = np.load(self.d / f"{name}.npz")
        x6 = z["x6"]; lbl = z["label"].astype(np.int64); spec = self.spec[name]
        _, SZ, SZw = x6.shape; cs = self.c
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge")
            lbl = np.pad(lbl, ((0, ph), (0, pw))); SZ, SZw = x6.shape[1:]
        if self.tr:
            t = random.randint(0, SZ - cs); l = random.randint(0, SZw - cs)
        else:
            t = (SZ - cs) // 2; l = (SZw - cs) // 2
        x = imnorm15(assemble15(x6, spec, t, l, cs)); lc = lbl[t:t + cs, l:l + cs]
        if self.tr:
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); lc = lc[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1, :].copy(); lc = lc[::-1, :].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2)).copy(); lc = np.rot90(lc, k).copy()
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lc))


@torch.no_grad()
def full_eval(model, names, dd, spec, dev, cs=448):
    model.eval(); tp = fp = fn = 0
    for name in names:
        z = np.load(Path(dd) / f"{name}.npz"); x6 = z["x6"]; lbl = z["label"]; sp = spec[name]
        _, SZ, SZw = x6.shape
        acc = np.zeros((3, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
        ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
        if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
        if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
        for t in ys:
            for l in xs:
                xb = torch.from_numpy(imnorm15(assemble15(x6, sp, t, l, cs))).unsqueeze(0).to(dev)
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
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_fuse")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr_names = man["train"]; te_names = man["test"]
    g_dicts = lambda ns: [{"county": n.split("_")[0], "idx": int(n.split("_")[1])} for n in ns]
    trc, _ = parallel_loadsplit_multitemp(g_dicts(tr_names), a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=16)
    tec, _ = parallel_loadsplit_multitemp(g_dicts(te_names), a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=8)
    spec = {**build_spec(trc), **build_spec(tec)}
    tr_names = [n for n in tr_names if n in spec]; te_names = [n for n in te_names if n in spec]
    print(f"[dino-fuse] 15ch train={len(tr_names)} test={len(te_names)}", flush=True)

    trl = torch.utils.data.DataLoader(FuseDS(tr_names, a.data_dir, spec, a.crop, True),
                                      batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=15, unfreeze_last_n=a.unfreeze).to(a.device)
    print(f"  trainable={sum(q.numel() for q in model.parameters() if q.requires_grad)/1e6:.1f}M", flush=True)

    bc = np.zeros(3)
    for n in tr_names[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)
    bb = [q for nm, q in model.named_parameters() if q.requires_grad and nm.startswith("backbone")]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and not nm.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr}, {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
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
        f1 = full_eval(model, te_names, a.data_dir, spec, a.device, a.crop)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} loss={el/nb:.4f} 1m-F1={f1:.4f} (best {best:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[FINAL dino-fuse] best 1m-F1={best:.4f}", flush=True)
    print(f"  vs DINOv2-1m (6ch, 1m-only) 0.860 in-domain | next: cross-province (spectral may hurt)", flush=True)
    json.dump({"best_1m_f1": best}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
