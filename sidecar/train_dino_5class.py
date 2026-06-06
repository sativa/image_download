"""DINOv2-1m MULTI-CLASS (5 landform types) — per user: recognize more types, use full Gansu DLTB.

Input = 6ch @ 1m (Esri+Google RGB, ImageNet-norm). Labels = 5-class 1m from `make_5class_labels.py`
(1耕地 2园地 3林地 4草地 5其他, 0=nodata ignored). num_classes=6 (0 ignored in CE). Reports overall
accuracy (valid pixels) + cropland(耕地+园地) F1 (comparable to the binary 0.86) + per-class recall.
Trains on ALL 5000 c_1m cells. Same DinoUNet5ch backbone as the binary model.
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2

NCLS = 6  # set in main from --nclass (6 = 5-class; 13 = 12 first-level classes)
NAMES5 = ["nodata", "耕地", "园地", "林地", "草地", "其他"]
NAMES12 = ["nodata", "耕地", "园地", "林地", "草地", "商服", "工矿", "住宅", "公管", "特殊", "交通", "水域", "其他"]
CLASS_NAMES = NAMES5


class C1mDS5(torch.utils.data.Dataset):
    def __init__(self, names, dd, lab5, crop, training, rare_set=None, rare_min=5):
        self.n = names; self.d = Path(dd); self.l = Path(lab5); self.c = crop; self.tr = training
        self.rare = rare_set or set(); self.rmin = rare_min  # bias crops to classes >= rare_min

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        nm = self.n[i]
        x6 = np.load(self.d / f"{nm}.npz")["x6"]; lbl = np.load(self.l / f"{nm}.npy").astype(np.int64)
        cs = self.c; _, SZ, SZw = x6.shape
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge"); lbl = np.pad(lbl, ((0, ph), (0, pw))); SZ, SZw = x6.shape[1:]
        if self.tr and nm in self.rare and random.random() < 0.7 and (lbl >= self.rmin).any():
            ys, xs = np.where(lbl >= self.rmin); k = random.randrange(len(ys))   # center crop on a rare-class pixel
            t = int(np.clip(ys[k] - cs // 2, 0, SZ - cs)); l = int(np.clip(xs[k] - cs // 2, 0, SZw - cs))
        elif self.tr:
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
def full_eval(model, names, dd, lab5, dev, cs=448):
    model.eval()
    correct = total = 0; tp = fp = fn = 0
    per_tp = np.zeros(NCLS); per_n = np.zeros(NCLS)
    for nm in names:
        x6 = np.load(Path(dd) / f"{nm}.npz")["x6"]; lbl = np.load(Path(lab5) / f"{nm}.npy")
        _, SZ, SZw = x6.shape
        acc = np.zeros((NCLS, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
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
        pred = acc.argmax(0); v = lbl > 0
        correct += int((pred[v] == lbl[v]).sum()); total += int(v.sum())
        cp = (pred == 1) | (pred == 2); cg = ((lbl == 1) | (lbl == 2)) & v; cpi = cp & v
        tp += int((cpi & cg).sum()); fp += int((cpi & ~cg & v).sum()); fn += int((~cp & cg).sum())
        for c in range(1, NCLS):
            m = (lbl == c)
            per_tp[c] += int(((pred == c) & m).sum()); per_n[c] += int(m.sum())
    oa = correct / max(total, 1)
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); cf1 = 2 * pr * rc / (pr + rc + 1e-9)
    recall = {CLASS_NAMES[c]: (per_tp[c] / per_n[c] if per_n[c] else 0) for c in range(1, NCLS)}
    return oa, cf1, recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--lab5", default="/mnt/sda/zf/landform/data/c_1m_label5")
    p.add_argument("--nclass", type=int, default=6, help="6=5-class; 13=12 first-level classes")
    p.add_argument("--rare-oversample", type=int, default=1, help="oversample factor for cells with rare classes (>=5)")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_5class")
    p.add_argument("--crop", type=int, default=448); p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3); p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    global NCLS, CLASS_NAMES
    NCLS = a.nclass; CLASS_NAMES = NAMES12 if a.nclass == 13 else NAMES5
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr = [n for n in man["train"] if (Path(a.lab5) / f"{n}.npy").exists()]
    te = [n for n in man["test"] if (Path(a.lab5) / f"{n}.npy").exists()]
    rare_set = set()
    if a.rare_oversample > 1:
        for n in tr:
            if (np.load(Path(a.lab5) / f"{n}.npy") >= 5).any():
                rare_set.add(n)
        tr = tr + [n for n in tr if n in rare_set] * (a.rare_oversample - 1)  # repeat rare cells
        print(f"[5cls] rare(>=5) cells {len(rare_set)} oversampled x{a.rare_oversample}", flush=True)
    print(f"[5cls] train={len(tr)} test={len(te)}", flush=True)

    trl = torch.utils.data.DataLoader(C1mDS5(tr, a.data_dir, a.lab5, a.crop, True, rare_set=rare_set),
                                      batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=NCLS, in_channels=6, unfreeze_last_n=a.unfreeze).to(a.device)
    bc = np.zeros(NCLS)
    for n in tr[:400]:
        bc += np.bincount(np.load(Path(a.lab5) / f"{n}.npy").ravel(), minlength=NCLS)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * (NCLS - 1)
    cwt = torch.from_numpy(cw).to(a.device)
    print(f"  class px%: " + " ".join(f"{CLASS_NAMES[i]}={bc[i]/bc.sum()*100:.1f}" for i in range(NCLS)), flush=True)
    bb = [q for nm, q in model.named_parameters() if q.requires_grad and nm.startswith("backbone")]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and not nm.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr}, {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs); scaler = torch.amp.GradScaler("cuda")
    best = -1
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for x, y in trl:
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(x)
                if lg.shape[-2:] != y.shape[-2:]:
                    lg = F.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(lg.float(), y, weight=cwt, ignore_index=0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        oa, cf1, rec = full_eval(model, te, a.data_dir, a.lab5, a.device, a.crop)
        score = oa
        if score > best:
            best = score; torch.save(model.state_dict(), out / "best.pt")
        recs = " ".join(f"{k}{v:.2f}" for k, v in rec.items())
        print(f"  ep{ep+1}/{a.epochs} OA={oa:.4f} cropF1={cf1:.4f} | {recs} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[5cls] best OA={best:.4f} | compare: binary cropland-F1 0.86", flush=True)
    json.dump({"best_oa": best}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
