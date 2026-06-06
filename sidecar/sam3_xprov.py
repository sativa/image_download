"""SAM3 + OBIA CROSS-PROVINCE test (Gansu-trained -> Shanxi/Changzhi).

The paper's value proposition: trained spectral models collapse cross-province (argmax F1 0.24),
but SAM3's parcel geometry is domain-invariant and should generalize. Train the per-parcel RF on
Gansu, then run SAM3 zero-shot parcel seg + RF classify on the 160 Changzhi cells.

Reports (vs the Changzhi 10m DLTB label):
  - zero-shot all-parcels 10m-F1  (pure SAM3 geometry recall — does it find Shanxi fields?)
  - SAM3 + Gansu-RF 10m-F1        (does the Gansu per-parcel classifier transfer?)
Compare to the trained 10m model cross-province: argmax 0.236 / best-thr 0.617.
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_route_a import build_spec, EXTRA_YEARS
from fast_load_multitemp import parallel_loadsplit_multitemp
from sam3_field_seg import load_processor
from sam3_obia import instances_10m, f1_counts, f1
from sam3_obia_rf import extract_cell


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
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_xprov")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Gansu training data (10m spec) ---
    man = json.loads((Path(a.gansu_dir) / "manifest.json").read_text())
    g_all = [n for n in man["train"] if (Path(a.gansu_dir) / f"{n}.npz").exists()]
    tr_names = g_all[::max(1, len(g_all) // a.n_train)][:a.n_train]
    g_dicts = [{"county": n.split("_")[0], "idx": int(n.split("_")[1])} for n in tr_names]
    gc, _ = parallel_loadsplit_multitemp(g_dicts, a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=16)
    g_spec = build_spec(gc); g_by = {c["name"]: c for c in gc}
    print(f"[xprov] Gansu 10m spec {len(g_spec)} cells ({time.time()-t0:.0f}s)", flush=True)

    proc = load_processor(a.weights, a.device, a.conf)
    print(f"[xprov] SAM3 loaded ({time.time()-t0:.0f}s)", flush=True)

    # --- train RF on Gansu parcels ---
    Xtr, ytr = [], []
    for i, n in enumerate(tr_names):
        if n not in g_spec: continue
        f, l, _, _ = extract_cell(proc, n, a.gansu_dir, g_spec[n], g_by[n]["label"], a.prompt, a.tile, a.conf, False)
        if len(f): Xtr.append(f); ytr.append(l)
        if (i + 1) % 50 == 0: print(f"  gansu extract {i+1}/{len(tr_names)} ({time.time()-t0:.0f}s)", flush=True)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    print(f"[xprov] RF trained on {len(ytr)} Gansu parcels (crop {ytr.mean():.2f}) ({time.time()-t0:.0f}s)", flush=True)

    # --- Changzhi 10m spec + labels from pkl ---
    # Trusted internal artifact: this project's own Changzhi 10m cell cache (built by changzhi_fuse.py
    # on .250), not external input — safe to unpickle.
    cz = pickle.load(open(a.cz_pkl, "rb"))
    cz_spec = build_spec(cz); cz_lbl = {c["name"]: c["label"] for c in cz}
    cz_names = [c["name"] for c in cz if (Path(a.cz_1m_dir) / f'{c["name"]}.npz').exists()]
    print(f"[xprov] Changzhi {len(cz_names)} cells with 1m+10m ({time.time()-t0:.0f}s)", flush=True)

    ZS = [0, 0, 0]; RF = [0, 0, 0]
    for i, name in enumerate(cz_names):
        sp = cz_spec[name]; lbl = cz_lbl[name]; Hs, Ws = sp.shape[1:]
        z = np.load(Path(a.cz_1m_dir) / f"{name}.npz"); rgb = np.ascontiguousarray(z["x6"][0:3].transpose(1, 2, 0))
        ws = instances_10m(proc, rgb, a.prompt, a.tile, Hs, Ws)

        def paint(sel):
            cov = np.zeros((Hs, Ws), np.float32)
            for j in np.where(sel)[0]:
                cov = np.maximum(cov, ws[j])
            return cov >= 0.5

        if ws:
            feats = np.array([(sp * w[None]).sum((1, 2)) / (w.sum() + 1e-9) for w in ws]).reshape(-1, 9)
            prob = rf.predict_proba(feats)[:, 1]
            zs = paint(np.ones(len(ws), bool)); rfm = paint(prob > 0.5)
        else:
            zs = rfm = np.zeros((Hs, Ws), bool)
        for acc, pr in ((ZS, zs), (RF, rfm)):
            c = f1_counts(pr, lbl); acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]
        if (i + 1) % 40 == 0: print(f"  changzhi {i+1}/{len(cz_names)} ({time.time()-t0:.0f}s)", flush=True)

    zs = f1(*ZS); rfr = f1(*RF)
    print(f"\n[xprov] === CROSS-PROVINCE (Gansu->Changzhi, {len(cz_names)} cells) ===", flush=True)
    print(f"  SAM3 zero-shot all-parcels: 10m-F1={zs[0]:.4f} (P{zs[1]:.2f}/R{zs[2]:.2f})", flush=True)
    print(f"  SAM3 + Gansu-RF OBIA:       10m-F1={rfr[0]:.4f} (P{rfr[1]:.2f}/R{rfr[2]:.2f})", flush=True)
    print(f"  compare: trained 10m model cross-province argmax 0.236 / best-thr 0.617", flush=True)
    print(f"           (in-domain Gansu: SAM3+RF 0.602, trained UNet 0.838)", flush=True)
    json.dump({"n_cells": len(cz_names), "zero_shot_10m_f1": zs[0], "zs_p": zs[1], "zs_r": zs[2],
               "rf_10m_f1": rfr[0], "rf_p": rfr[1], "rf_r": rfr[2]},
              open(out / "xprov_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
