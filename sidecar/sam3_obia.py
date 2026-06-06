"""SAM3 parcels + 10m-NDVI OBIA -> cropland (the user's core 1m-PRIMARY architecture).

Pipeline: SAM3 zero-shot segments field parcels on 1m Esri RGB -> for each parcel, area-aggregate
the 9-ch 10m spectral/NDVI stack -> classify parcel cropland by an NDVI threshold (unsupervised).
This is the precision fix for zero-shot SAM3: bare/terraced hillsides that SAM3 calls "field" get
rejected by low NDVI, while real cropland (high seasonal NDVI) is kept.

Reports, per NDVI threshold, BOTH 10m-aggregated F1 (directly comparable to the 10m baseline 0.853)
and 1m-F1, plus the zero-shot (all-parcels) baseline. c_1m 1m grid is exactly 10x the 10m grid.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_route_a import build_spec, EXTRA_YEARS
from fast_load_multitemp import parallel_loadsplit_multitemp
from sam3_field_seg import load_processor


def instances_10m(proc, rgb, prompt, tile, Hs, Ws):
    """Run SAM3; return list of per-instance 10m coverage maps (Hs,Ws) float in [0,1].
    Robust to any 1m:10m ratio (area-pools each instance to the window's 10m sub-grid)."""
    H, W = rgb.shape[:2]
    wins = [(0, 0, H, W)]
    if tile and (H > tile or W > tile):
        wins = [(y, x, min(y + tile, H), min(x + tile, W))
                for y in range(0, H, tile) for x in range(0, W, tile)]
    ws = []
    for (y0, x0, y1, x1) in wins:
        st = proc.set_image(Image.fromarray(rgb[y0:y1, x0:x1]))
        st = proc.set_text_prompt(prompt=prompt, state=st)
        m = st.get("masks")
        if m is None or m.numel() == 0:
            continue
        mi = m.squeeze(1).float().cpu()  # [N,h,w]
        r0 = int(round(y0 / H * Hs)); r1 = int(round(y1 / H * Hs))
        c0 = int(round(x0 / W * Ws)); c1 = int(round(x1 / W * Ws))
        th, tw = max(1, r1 - r0), max(1, c1 - c0)
        sub = F.interpolate(mi.unsqueeze(1), size=(th, tw), mode="area")[:, 0].numpy()  # [N,th,tw]
        for s in sub:
            if s.sum() <= 0:
                continue
            wfull = np.zeros((Hs, Ws), np.float32)
            wfull[r0:r1, c0:c1] = s
            ws.append(wfull)
    return ws


def up1m(c10, H, W):
    """Upsample a 10m bool mask to the 1m (H,W) grid (nearest)."""
    t = torch.from_numpy(c10.astype(np.float32))[None, None]
    return F.interpolate(t, size=(H, W), mode="nearest")[0, 0].numpy() > 0.5


def f1_counts(pred, lbl):
    v = lbl > 0; gt = (lbl == 1) & v; pi = pred & v
    return int((pi & gt).sum()), int((pi & ~gt & v).sum()), int((~pred & gt).sum())


def f1(tp, fp, fn):
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--n-cells", type=int, default=8)
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_obia")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te_all = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    step = max(1, len(te_all) // a.n_cells)
    names = te_all[::step][:a.n_cells]
    te_dicts = [{"county": n.split("_")[0], "idx": int(n.split("_")[1])} for n in names]
    print(f"[obia] {len(names)} cells | prompt='{a.prompt}' tile={a.tile} conf={a.conf}", flush=True)

    t0 = time.time()
    tec, _ = parallel_loadsplit_multitemp(te_dicts, a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=8)
    spec = build_spec(tec); te_by = {c["name"]: c for c in tec}
    print(f"[obia] loaded 10m spec for {len(spec)} cells ({time.time()-t0:.0f}s)", flush=True)

    proc = load_processor(a.weights, a.device, a.conf)
    print(f"[obia] SAM3 loaded ({time.time()-t0:.0f}s)", flush=True)

    thrs = np.round(np.linspace(-1.0, 0.8, 19), 3)
    A10 = {t: [0, 0, 0] for t in thrs}; A1 = {t: [0, 0, 0] for t in thrs}
    ZS10 = [0, 0, 0]; ZS1 = [0, 0, 0]  # zero-shot (all parcels = cropland)

    for ci, name in enumerate(names):
        if name not in spec:
            print(f"  skip {name} (no 10m)", flush=True); continue
        z = np.load(Path(a.data_dir) / f"{name}.npz")
        rgb = np.ascontiguousarray(z["x6"][0:3].transpose(1, 2, 0)); lbl1 = z["label"]
        sp = spec[name]; Hs, Ws = sp.shape[1:]; lbl10 = te_by[name]["label"]
        tc = time.time()
        ws = instances_10m(proc, rgb, a.prompt, a.tile, Hs, Ws)
        scores = np.array([float((sp[4:9] * w[None]).sum() / (w.sum() + 1e-9) / 5) for w in ws])

        def cov_mask(sel):
            cov = np.zeros((Hs, Ws), np.float32)
            for w in (ws[i] for i in np.where(sel)[0]):
                cov = np.maximum(cov, w)
            return cov >= 0.5

        H1, W1 = lbl1.shape
        zs10 = cov_mask(np.ones(len(ws), bool)); zs1 = up1m(zs10, H1, W1)
        for acc, pr, lb in ((ZS10, zs10, lbl10), (ZS1, zs1, lbl1)):
            c = f1_counts(pr, lb); acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]
        for t in thrs:
            c10 = cov_mask(scores > t); c1 = up1m(c10, H1, W1)
            for acc, pr, lb in ((A10[t], c10, lbl10), (A1[t], c1, lbl1)):
                c = f1_counts(pr, lb); acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]
        srange = f"[{scores.min():.2f},{scores.max():.2f}]" if len(scores) else "[empty]"
        print(f"  [{ci+1}/{len(names)}] {name}: {len(ws)} parcels, score{srange} ({time.time()-tc:.0f}s)", flush=True)

    zf10 = f1(*ZS10); zf1 = f1(*ZS1)
    print(f"\n[obia] zero-shot (all parcels): 10m-F1={zf10[0]:.4f} (P{zf10[1]:.2f}/R{zf10[2]:.2f}) | 1m-F1={zf1[0]:.4f}", flush=True)
    print(f"[obia] NDVI-threshold sweep:", flush=True)
    best = (-1, None)
    rows = []
    for t in thrs:
        a10 = f1(*A10[t]); a1 = f1(*A1[t])
        rows.append({"thr": float(t), "f1_10m": a10[0], "p_10m": a10[1], "r_10m": a10[2], "f1_1m": a1[0]})
        flag = ""
        if a10[0] > best[0]: best = (a10[0], t); flag = " *"
        print(f"  thr={t:+.2f}: 10m-F1={a10[0]:.4f} (P{a10[1]:.2f}/R{a10[2]:.2f}) | 1m-F1={a1[0]:.4f}{flag}", flush=True)
    print(f"\n[obia] BEST 10m-F1={best[0]:.4f} @ NDVI-thr={best[1]:+.2f}", flush=True)
    print(f"  compare: 10m-only baseline 0.853 | SAM3 zero-shot all-parcels 10m-F1={zf10[0]:.4f}", flush=True)
    json.dump({"cells": names, "prompt": a.prompt, "zero_shot_10m_f1": zf10[0], "zero_shot_1m_f1": zf1[0],
               "best_10m_f1": best[0], "best_thr": best[1], "sweep": rows},
              open(out / "obia_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
