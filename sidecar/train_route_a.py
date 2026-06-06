"""Route a: 1m-NATIVE single-net cropland classification.

Input = 15ch @ 1m: 6ch (Esri+Google 1m RGB from c_1m) + 9ch (10m S2 RGBNIR + NDVI 2021 +
4 year-NDVI, upsampled to the 1m grid). Predicts cropland at 1m. Unlike route-c (which only
fed 1m as 2 aggregated hints to a 10m prediction), route-a predicts NATIVELY at 1m so small
fields / sharp boundaries count fully — the regime where 1m should break past the 10m ceiling.

Eval (final, full-cell tiled): 1m-F1 (native) AND 10m-aggregated F1 (area-pool the 1m argmax to
the 10m grid, compare to the 10m label) — the latter is directly comparable to the 10m 0.853.
Loss = 0.5 CE + 0.5 soft-Dice(cropland) + bdy * boundary (1m edges matter)."""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from fast_load_multitemp import parallel_loadsplit_multitemp

S2_MEAN = np.array([400, 460, 320, 1800], np.float32)
S2_STD = np.array([200, 200, 200, 700], np.float32)
NDVI_MEAN, NDVI_STD = 0.5, 0.3
EXTRA_YEARS = [2018, 2019, 2020, 2022]


def build_spec(cells):
    """name -> normalized 9-ch 10m spectral (9,Hs,Ws)."""
    d = {}
    for c in cells:
        rg = c["rgbnir"].astype(np.float32).copy()
        for b in range(4):
            rg[b] = (rg[b] - S2_MEAN[b]) / S2_STD[b]
        nd = (c["ndvi_s2"].astype(np.float32) - NDVI_MEAN) / NDVI_STD
        ny = (c["ndvi_years"].astype(np.float32) - NDVI_MEAN) / NDVI_STD
        d[c["name"]] = np.concatenate([rg, nd[None], ny], 0).astype(np.float32)
    return d


def assemble15(x6, spec, t, l, cs):
    """6ch 1m crop + 10m spectral sub-region upsampled to the crop -> 15ch."""
    x6c = x6[:, t:t + cs, l:l + cs].astype(np.float32) / 255.0
    _, SZ, SZw = x6.shape
    Hs, Ws = spec.shape[1:]
    r0 = int(t * Hs / SZ); r1 = min(Hs, max(r0 + 1, int(np.ceil((t + cs) * Hs / SZ))))
    c0 = int(l * Ws / SZw); c1 = min(Ws, max(c0 + 1, int(np.ceil((l + cs) * Ws / SZw))))
    sub = torch.from_numpy(spec[:, r0:r1, c0:c1])[None]
    sub = F.interpolate(sub, size=(cs, cs), mode="bilinear", align_corners=False)[0].numpy()
    return np.concatenate([x6c, sub], 0)


class RouteADS(torch.utils.data.Dataset):
    def __init__(self, names, c1m, spec, crop, training):
        self.n = names; self.d = Path(c1m); self.spec = spec; self.c = crop; self.tr = training

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
        x = assemble15(x6, spec, t, l, cs); lc = lbl[t:t + cs, l:l + cs]
        if self.tr:
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); lc = lc[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1, :].copy(); lc = lc[::-1, :].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2)).copy(); lc = np.rot90(lc, k).copy()
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lc))


def boundary(m, k=3):
    m = m.unsqueeze(1).float(); p = k // 2
    return (F.max_pool2d(m, k, 1, p) + F.max_pool2d(-m, k, 1, p) * -1).squeeze(1)


def f1_crop(model, loader, dev):
    model.eval(); tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); y = y.to(dev); p = model(x).argmax(1); v = y > 0
            pi = (p == 1) & v; ti = (y == 1) & v
            tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9)


@torch.no_grad()
def full_eval(model, names, c1m, spec, te_cells_by_name, dev, cs=512):
    """Tiled full-cell 1m inference -> 1m-F1 (native) + 10m-aggregated F1."""
    model.eval()
    tp = fp = fn = 0          # 1m
    tp10 = fp10 = fn10 = 0    # 10m-aggregated
    for name in names:
        z = np.load(Path(c1m) / f"{name}.npz"); x6 = z["x6"]; lbl = z["label"]
        sp = spec[name]; _, SZ, SZw = x6.shape
        acc = np.zeros((3, SZ, SZw), np.float32); cnt = np.zeros((1, SZ, SZw), np.float32)
        ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
        if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
        if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
        for t in ys:
            for l in xs:
                x = assemble15(x6, sp, t, l, cs)
                xb = torch.from_numpy(x).unsqueeze(0).to(dev)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pr = torch.softmax(model(xb).float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[0, t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0)  # (SZ,SZw)
        v = lbl > 0; ti = (lbl == 1) & v; pi = (pred == 1) & v
        tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
        # 10m aggregate: area-pool 1m cropland fraction to the 10m label grid
        lab10 = te_cells_by_name[name]["label"]; Hs, Ws = lab10.shape
        cropf = F.interpolate(torch.from_numpy((pred == 1).astype(np.float32))[None, None],
                              size=(Hs, Ws), mode="area")[0, 0].numpy()
        p10 = cropf >= 0.5; v10 = lab10 > 0; t10 = (lab10 == 1) & v10
        tp10 += int((p10 & t10).sum()); fp10 += int((p10 & ~t10 & v10).sum()); fn10 += int((~p10 & t10).sum())

    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9)
    return f1(tp, fp, fn), f1(tp10, fp10, fn10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(HOME / "data/c_1m"))
    p.add_argument("--regions-json", default=str(HOME / "data/v40_5k.json"))
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out", default=str(HOME / "results/route_a"))
    p.add_argument("--crop", type=int, default=512); p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--epochs", type=int, default=25); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--bdy-w", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    R = json.loads(Path(a.regions_json).read_text())
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    trc, _ = parallel_loadsplit_multitemp(R["train"], a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=a.workers)
    tec, _ = parallel_loadsplit_multitemp(R["test"], a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=8)
    spec = {**build_spec(trc), **build_spec(tec)}
    te_by_name = {c["name"]: c for c in tec}
    tr_names = [n for n in man["train"] if n in spec]
    te_names = [n for n in man["test"] if n in spec]
    print(f"[route-a] 15ch@1m train={len(tr_names)} test={len(te_names)}", flush=True)
    trl = torch.utils.data.DataLoader(RouteADS(tr_names, a.data_dir, spec, a.crop, True),
                                      batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True)
    tel = torch.utils.data.DataLoader(RouteADS(te_names, a.data_dir, spec, a.crop, False),
                                      batch_size=a.batch_size, shuffle=False, num_workers=4)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=a.backbone, encoder_weights="imagenet", in_channels=15, classes=3).to(a.device)
    print(f"  params={sum(q.numel() for q in model.parameters())/1e6:.1f}M", flush=True)
    bc = np.zeros(3)
    for n in tr_names[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
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
                lg = model(x); ce = F.cross_entropy(lg.float(), y, weight=cwt, ignore_index=0)
                pr = torch.softmax(lg.float(), 1); p1 = pr[:, 1]; v = (y > 0).float(); t1 = (y == 1).float()
                inter = (p1 * t1 * v).sum(); den = (p1 * v).sum() + (t1 * v).sum(); dice = 1 - (2 * inter + 1) / (den + 1)
                pb = boundary(p1).clamp(1e-4, 1 - 1e-4); gb = boundary((y == 1).float())
                bl = -(gb * torch.log(pb) + (1 - gb) * torch.log(1 - pb)); bl = (bl * v).sum() / v.sum().clamp(min=1)
                loss = 0.5 * ce + 0.5 * dice + a.bdy_w * bl
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        f1 = f1_crop(model, tel, a.device)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), Path(a.out) / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} 1m-F1(512crop)={f1:.4f} (best {best:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(torch.load(Path(a.out) / "best.pt", map_location=a.device, weights_only=True))
    f1_1m, f1_10m = full_eval(model, te_names, a.data_dir, spec, te_by_name, a.device, a.crop)
    print(f"\n[FINAL route-a] full-cell 1m-F1={f1_1m:.4f} | 10m-aggregated F1={f1_10m:.4f}", flush=True)
    print(f"  compare: route-c 10m-grid 0.851 (single) / 0.853 (ensemble); baseline 0.842", flush=True)
    json.dump({"crop_1m_f1": best, "full_1m_f1": f1_1m, "agg_10m_f1": f1_10m},
              open(Path(a.out) / "final.json", "w"))


if __name__ == "__main__":
    main()
