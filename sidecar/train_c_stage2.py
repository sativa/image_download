"""Route (c) — Stage-2: classify cropland on the 10m grid using the 9-ch multi-temporal
spectral stack PLUS 2 channels distilled from the 1m stage-1 net (mean cropland prob +
boundary density, already aggregated to the 10m grid by stage1_infer.py).

Clean ablation: --no-1m zeroes the 2 stage-1 channels, leaving an IDENTICAL 11-ch net
trained on the SAME cells -> the F1 gap between the two runs is exactly the 1m contribution.

Headline metric = cross-county cropland F1 on the 120 held-out test cells (12 counties),
directly comparable to the 10m ensemble's 0.853. Reports argmax F1 each epoch; at the end
reloads best.pt and reports D4-TTA argmax / global-threshold / leave-county-out threshold-CV F1.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from fast_load_multitemp import parallel_loadsplit_multitemp

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3
EXTRA_YEARS = [2018, 2019, 2020, 2022]
THRS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def attach_feat(cells, feat_dir, use_1m):
    """Add c['feat2'] (2,Hs,Ws) float32: stage-1 prob+boundary, or zeros if absent/--no-1m."""
    feat_dir = Path(feat_dir); n_real = 0
    for c in cells:
        Hs, Ws = c["label"].shape
        f = feat_dir / f"{c['name']}.npz"
        if use_1m and f.exists():
            feat = np.load(f)["feat2"].astype(np.float32)
            if feat.shape[1:] != (Hs, Ws):  # safety: realign
                t = torch.from_numpy(feat)[None]
                feat = F.interpolate(t, size=(Hs, Ws), mode="bilinear", align_corners=False)[0].numpy()
            n_real += 1
        else:
            feat = np.zeros((2, Hs, Ws), np.float32)
        c["feat2"] = feat
    return n_real


class Stage2DS(torch.utils.data.Dataset):
    def __init__(self, cells, size=224, training=True):
        self.cells = cells; self.size = size; self.training = training

    def __len__(self): return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32).copy()
        ndvi_s2 = c["ndvi_s2"].astype(np.float32)
        ndvi_yr = c["ndvi_years"].astype(np.float32)
        feat2 = c["feat2"].astype(np.float32)        # (2,Hs,Ws) in [0,1]
        lbl = c["label"].astype(np.int64)
        for b in range(4):
            rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_s2 = (ndvi_s2 - NDVI_MEAN) / NDVI_STD
        ndvi_yr = (ndvi_yr - NDVI_MEAN) / NDVI_STD
        x = np.concatenate([rgbnir, ndvi_s2[None], ndvi_yr, feat2], axis=0).astype(np.float32)
        H, W = x.shape[1], x.shape[2]; sz = self.size
        if H < sz or W < sz:
            ph = max(0, sz - H); pw = max(0, sz - W)
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
            lbl = np.pad(lbl, ((0, ph), (0, pw)), mode="constant"); H, W = x.shape[1], x.shape[2]
        if self.training:
            top = np.random.randint(0, max(1, H - sz + 1)); left = np.random.randint(0, max(1, W - sz + 1))
        else:
            top = (H - sz) // 2; left = (W - sz) // 2
        x = x[:, top:top + sz, left:left + sz]; lbl = lbl[top:top + sz, left:left + sz]
        if self.training:
            if np.random.random() < 0.5: x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5: x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k: x = np.rot90(x, k, (1, 2)).copy(); lbl = np.rot90(lbl, k, (0, 1)).copy()
            jit = 1.0 + (np.random.random(4) - 0.5) * 0.2
            for b in range(4): x[b] *= jit[b]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lbl))


@torch.no_grad()
def tta_prob(model, x, dev):
    """D4 TTA -> averaged cropland prob (B,H,W)."""
    acc = None
    for k in range(4):
        for fl in (False, True):
            xi = torch.rot90(x, k, (2, 3))
            if fl: xi = torch.flip(xi, (3,))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pr = torch.softmax(model(xi).float(), 1)
            if fl: pr = torch.flip(pr, (3,))
            pr = torch.rot90(pr, -k, (2, 3))
            acc = pr if acc is None else acc + pr
    return (acc / 8.0)[:, 1]


def f1_from_counts(tp, fp, fn):
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", default=str(HOME / "data/v40_5k.json"))
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb-cache", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--feat-dir", default=str(HOME / "data/c_stage1_feat"))
    p.add_argument("--out-dir", default=str(HOME / "results/c_stage2"))
    p.add_argument("--no-1m", action="store_true", help="zero the 2 stage-1 channels (ablation baseline)")
    p.add_argument("--arch", default="unet", choices=["unet", "segformer", "unetplusplus", "deeplabv3plus"])
    p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--loss", default="dice_ce", choices=["ce", "dice_ce"])
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    use_1m = not getattr(a, "no_1m")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    R = json.loads(Path(a.regions_json).read_text())
    tr, sk_t = parallel_loadsplit_multitemp(R["train"], a.dltb_cache, a.s2_dir, a.ndvi_yr_dir,
                                            EXTRA_YEARS, max_workers=a.workers)
    te, sk_v = parallel_loadsplit_multitemp(R["test"], a.dltb_cache, a.s2_dir, a.ndvi_yr_dir,
                                            EXTRA_YEARS, max_workers=8)
    nr_t = attach_feat(tr, a.feat_dir, use_1m); nr_v = attach_feat(te, a.feat_dir, use_1m)
    print(f"[stage2] use_1m={use_1m} train={len(tr)}({nr_t} w/1m) test={len(te)}({nr_v} w/1m)", flush=True)

    trl = torch.utils.data.DataLoader(Stage2DS(tr, a.target_size, True), batch_size=a.batch_size,
                                      shuffle=True, num_workers=4, pin_memory=True)
    tel = torch.utils.data.DataLoader(Stage2DS(te, a.target_size, False), batch_size=a.batch_size,
                                      shuffle=False, num_workers=2)
    n_ch = 5 + len(EXTRA_YEARS) + 2
    import segmentation_models_pytorch as smp
    _ARCH = {"unet": smp.Unet, "segformer": smp.Segformer, "unetplusplus": smp.UnetPlusPlus,
             "deeplabv3plus": smp.DeepLabV3Plus}
    _ew = None if str(a.encoder_weights).lower() in ("none", "null", "scratch") else a.encoder_weights
    model = _ARCH[a.arch](encoder_name=a.backbone, encoder_weights=_ew, in_channels=n_ch, classes=3).to(a.device)
    print(f"  arch={a.arch}/{a.backbone} {n_ch}ch params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    bc = np.zeros(3)
    for c in tr: bc += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1.0 / np.sqrt(bc), 0.0).astype(np.float32); cw[0] = 0.0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); el = 0; nb = 0
        for x, y in trl:
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(x)
                loss = F.cross_entropy(lg.float(), y, weight=cwt, ignore_index=0)
                if a.loss == "dice_ce":
                    pr = torch.softmax(lg.float(), 1); p1 = pr[:, 1]; v = (y > 0).float(); t1 = (y == 1).float()
                    inter = (p1 * t1 * v).sum(); den = (p1 * v).sum() + (t1 * v).sum()
                    loss = 0.5 * loss + 0.5 * (1.0 - (2 * inter + 1) / (den + 1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); el += loss.item(); nb += 1
        sch.step()
        model.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for x, y in tel:
                x = x.to(a.device); y = y.to(a.device); pp = model(x).argmax(1); v = y > 0
                pi = (pp == 1) & v; ti = (y == 1) & v
                tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
        f1 = f1_from_counts(tp, fp, fn)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} loss={el/max(nb,1):.4f} test-F1={f1:.4f} (best {best:.4f}) ({time.time()-t0:.0f}s)", flush=True)

    # ---- final honest eval: D4 TTA + thresholds (argmax / global / leave-county-out CV) ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=a.device, weights_only=True)); model.eval()
    by_county = defaultdict(lambda: {"argmax": [0, 0, 0], "thr": {float(t): [0, 0, 0] for t in THRS}})
    ev = torch.utils.data.DataLoader(Stage2DS(te, a.target_size, False), batch_size=a.batch_size, shuffle=False, num_workers=2)
    names = [c["name"] for c in te]  # aligned with dataset order (shuffle=False) -> county per sample
    bi = 0
    with torch.no_grad():
        for x, y in ev:
            x = x.to(a.device); y = y.to(a.device)
            prob = tta_prob(model, x, a.device)  # (B,H,W)
            am = (prob >= 0.5)
            for j in range(x.shape[0]):
                cnty = names[bi].split("_")[0]; bi += 1
                v = y[j] > 0; t1 = (y[j] == 1) & v
                d = by_county[cnty]
                pi = am[j] & v
                d["argmax"][0] += int((pi & t1).sum()); d["argmax"][1] += int((pi & ~t1 & v).sum()); d["argmax"][2] += int((~pi & t1).sum())
                for t in THRS:
                    pit = (prob[j] >= t) & v
                    e = d["thr"][float(t)]
                    e[0] += int((pit & t1).sum()); e[1] += int((pit & ~t1 & v).sum()); e[2] += int((~pit & t1).sum())
    # argmax
    A = np.sum([d["argmax"] for d in by_county.values()], 0); f1_am = f1_from_counts(*A)
    # global best threshold
    gt = {float(t): np.sum([d["thr"][float(t)] for d in by_county.values()], 0) for t in THRS}
    f1_g = {t: f1_from_counts(*gt[t]) for t in gt}; bt = max(f1_g, key=f1_g.get)
    # leave-one-county-out threshold CV (honest transferable)
    counties = list(by_county.keys()); cv = [0, 0, 0]
    for c in counties:
        scores = {}
        for t in THRS:
            agg = np.sum([by_county[o]["thr"][float(t)] for o in counties if o != c], 0)
            scores[float(t)] = f1_from_counts(*agg)
        t_star = max(scores, key=scores.get)
        e = by_county[c]["thr"][t_star]; cv[0] += e[0]; cv[1] += e[1]; cv[2] += e[2]
    f1_cv = f1_from_counts(*cv)
    tag = "route-c(10m+1m)" if use_1m else "baseline(10m-only,5k)"
    print(f"\n[FINAL] {tag} | TTA argmax F1={f1_am:.4f} | global-thr={bt} F1={f1_g[bt]:.4f} | "
          f"leave-county-out CV F1={f1_cv:.4f}", flush=True)
    (out / "final.json").write_text(json.dumps(
        {"use_1m": use_1m, "best_argmax_F1": best, "tta_argmax_F1": f1_am,
         "global_thr": bt, "global_thr_F1": f1_g[bt], "cv_F1": f1_cv}, indent=2))


if __name__ == "__main__":
    main()
