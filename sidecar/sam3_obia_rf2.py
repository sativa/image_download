"""SAM3 + ENRICHED-feature OBIA (per user request: add SAVI/EVI/... + richer zonal stats).

Upgrades over sam3_obia_rf.py (which used 9-ch MEAN only -> 0.602 in-domain / 0.649 cross-province):
  1. Spectral indices from the 4-band S2 (B,G,R,NIR): NDVI, SAVI, EVI, GNDVI, NDWI, MSAVI
     (research: EVI/SAVI top cropland indices) + the precomputed ndvi_s2 + 4-yr NDVI = 15 channels.
  2. Per-parcel zonal stats = MEAN + STD (+ area), not just mean (OBIA best practice — std captures
     within-parcel homogeneity: cropland is uniform, mixed/FP parcels are heterogeneous).
RF is tree-based (scale-invariant) so no normalization needed. One run reports in-domain (Gansu
cross-county) AND cross-province (Changzhi) so the enriched-feature gain is measured on both.

S2 band order verified [Blue,Green,Red,NIR]; reflectance is x10000 (divide for SAVI/EVI/MSAVI consts).
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from fast_load_multitemp import parallel_loadsplit_multitemp
from sam3_field_seg import load_processor
from sam3_obia import instances_10m, f1_counts, f1

EXTRA_YEARS = [2018, 2019, 2020, 2022]


def build_indices(cell):
    """cell -> (15, H, W) feature stack: 4 bands + 6 indices + ndvi_s2 + 4-yr NDVI."""
    rg = cell["rgbnir"].astype(np.float32)            # [B,G,R,NIR] raw (x10000 reflectance)
    B, G, R, NIR = rg[0], rg[1], rg[2], rg[3]
    rho = rg / 10000.0; b, g, r, nir = rho[0], rho[1], rho[2], rho[3]
    e = 1e-6
    ndvi = (nir - r) / (nir + r + e)
    savi = 1.5 * (nir - r) / (nir + r + 0.5)
    evi = np.clip(2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1 + e), -1.5, 1.5)
    gndvi = (nir - g) / (nir + g + e)
    ndwi = (g - nir) / (g + nir + e)
    msavi = (2 * nir + 1 - np.sqrt(np.clip((2 * nir + 1) ** 2 - 8 * (nir - r), 0, None))) / 2
    nd_s2 = cell["ndvi_s2"].astype(np.float32)
    ndy = cell["ndvi_years"].astype(np.float32)       # (4,H,W)
    return np.stack([B / 3000, G / 3000, R / 3000, NIR / 3000,
                     ndvi, savi, evi, gndvi, ndwi, msavi, nd_s2, *ndy], 0).astype(np.float32)


def agg(stack, w):
    """Per-parcel zonal stats: mean + std of each channel + parcel area -> (2K+1,)."""
    s = w.sum() + 1e-9
    mean = (stack * w[None]).sum((1, 2)) / s
    var = (stack ** 2 * w[None]).sum((1, 2)) / s - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    return np.concatenate([mean, std, [s]])


def extract(proc, rgb, stack, lbl10, prompt, tile, conf, keep_w):
    Hs, Ws = stack.shape[1:]
    ws = instances_10m(proc, rgb, prompt, tile, Hs, Ws)
    crop = (lbl10 == 1).astype(np.float32); valid = (lbl10 > 0).astype(np.float32)
    feats, labels = [], []
    for w in ws:
        feats.append(agg(stack, w))
        cf = (w * crop).sum() / ((w * valid).sum() + 1e-9)
        labels.append(1 if cf > 0.5 else 0)
    F = np.array(feats).reshape(len(feats), -1) if feats else np.zeros((0, 2 * stack.shape[0] + 1))
    return F, np.array(labels), (ws if keep_w else None)


def eval_set(proc, names, getrgb, getstack, getlbl, rf, prompt, tile, conf, tag, t0):
    """Run SAM3+RF over a cell set; return (zero-shot f1, rf f1) at 10m."""
    ZS = [0, 0, 0]; RF = [0, 0, 0]
    for i, n in enumerate(names):
        stack = getstack(n); lbl = getlbl(n); Hs, Ws = stack.shape[1:]
        F, _, ws = extract(proc, getrgb(n), stack, lbl, prompt, tile, conf, True)

        def paint(sel):
            cov = np.zeros((Hs, Ws), np.float32)
            for j in np.where(sel)[0]:
                cov = np.maximum(cov, ws[j])
            return cov >= 0.5
        zs = paint(np.ones(len(ws), bool)) if ws else np.zeros((Hs, Ws), bool)
        rfm = paint(rf.predict_proba(F)[:, 1] > 0.5) if len(F) else np.zeros((Hs, Ws), bool)
        for acc, pr in ((ZS, zs), (RF, rfm)):
            c = f1_counts(pr, lbl); acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]
        if (i + 1) % 40 == 0: print(f"  [{tag}] {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    return f1(*ZS), f1(*RF)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--gansu-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--cz-pkl", default="/mnt/sda/zf/landform/data/changzhi_cells.pkl")
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_obia_rf2")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    man = json.loads((Path(a.gansu_dir) / "manifest.json").read_text())
    sub = lambda lst, n: (lambda e: e[::max(1, len(e) // n)][:n])([x for x in lst if (Path(a.gansu_dir) / f"{x}.npz").exists()])
    tr_names = sub(man["train"], a.n_train); te_names = sub(man["test"], a.n_test)
    dicts = [{"county": n.split("_")[0], "idx": int(n.split("_")[1])} for n in tr_names + te_names]
    gc, _ = parallel_loadsplit_multitemp(dicts, a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=16)
    g_stack = {c["name"]: build_indices(c) for c in gc}; g_lbl = {c["name"]: c["label"] for c in gc}
    print(f"[obia2] Gansu indices {len(g_stack)} cells ({time.time()-t0:.0f}s)", flush=True)
    proc = load_processor(a.weights, a.device, a.conf)
    print(f"[obia2] SAM3 loaded ({time.time()-t0:.0f}s)", flush=True)

    Xtr, ytr = [], []
    for i, n in enumerate(tr_names):
        if n not in g_stack: continue
        z = np.load(Path(a.gansu_dir) / f"{n}.npz"); rgb = np.ascontiguousarray(z["x6"][0:3].transpose(1, 2, 0))
        F, l, _ = extract(proc, rgb, g_stack[n], g_lbl[n], a.prompt, a.tile, a.conf, False)
        if len(F): Xtr.append(F); ytr.append(l)
        if (i + 1) % 50 == 0: print(f"  train {i+1}/{len(tr_names)} ({time.time()-t0:.0f}s)", flush=True)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    print(f"[obia2] RF on {len(ytr)} parcels, {Xtr.shape[1]} feats (crop {ytr.mean():.2f}) ({time.time()-t0:.0f}s)", flush=True)

    g_rgb = lambda n: np.ascontiguousarray(np.load(Path(a.gansu_dir) / f"{n}.npz")["x6"][0:3].transpose(1, 2, 0))
    gz, grf = eval_set(proc, te_names, g_rgb, lambda n: g_stack[n], lambda n: g_lbl[n], rf, a.prompt, a.tile, a.conf, "gansu-test", t0)

    cz = pickle.load(open(a.cz_pkl, "rb"))  # trusted internal artifact (changzhi_fuse.py output)
    c_stack = {c["name"]: build_indices(c) for c in cz}; c_lbl = {c["name"]: c["label"] for c in cz}
    cz_names = [c["name"] for c in cz if (Path(a.cz_1m_dir) / f'{c["name"]}.npz').exists()]
    c_rgb = lambda n: np.ascontiguousarray(np.load(Path(a.cz_1m_dir) / f"{n}.npz")["x6"][0:3].transpose(1, 2, 0))
    czz, czrf = eval_set(proc, cz_names, c_rgb, lambda n: c_stack[n], lambda n: c_lbl[n], rf, a.prompt, a.tile, a.conf, "changzhi", t0)

    print(f"\n[obia2] === ENRICHED-FEATURE OBIA ({Xtr.shape[1]} feats: 15ch mean+std+area) ===", flush=True)
    print(f"  IN-DOMAIN (Gansu, {len(te_names)} cells):  zero-shot={gz[0]:.4f} | RF={grf[0]:.4f} (P{grf[1]:.2f}/R{grf[2]:.2f})", flush=True)
    print(f"  CROSS-PROVINCE (Changzhi, {len(cz_names)}):  zero-shot={czz[0]:.4f} | RF={czrf[0]:.4f} (P{czrf[1]:.2f}/R{czrf[2]:.2f})", flush=True)
    print(f"  vs 9ch-mean baseline: in-domain RF 0.602 | cross-province RF 0.649", flush=True)
    json.dump({"n_feats": int(Xtr.shape[1]), "indomain_zs": gz[0], "indomain_rf": grf[0],
               "xprov_zs": czz[0], "xprov_rf": czrf[0], "xprov_rf_p": czrf[1], "xprov_rf_r": czrf[2]},
              open(out / "obia_rf2_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
