"""SAM3 parcels + SUPERVISED per-parcel classifier (the proper OBIA).

The NDVI-threshold OBIA failed (SAM3's false positives are moderate-NDVI non-cropland that a
single threshold can't separate). This trains a RandomForest on the full 9-ch 10m feature vector
aggregated per SAM3 parcel -> DLTB-majority cropland label, then classifies test-cell parcels.
The 9-ch pixel classifier reaches 0.853 at 10m, so a multi-feature per-parcel RF should recover
much of that WHERE SAM3 found a parcel (recall is capped by SAM3's parcel coverage).

Reports 10m-F1 (vs 0.853 baseline) + 1m-F1, vs the zero-shot all-parcels baseline.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_route_a import build_spec, EXTRA_YEARS
from fast_load_multitemp import parallel_loadsplit_multitemp
from sam3_field_seg import load_processor
from sam3_obia import instances_10m, up1m, f1_counts, f1


def extract_cell(proc, name, dd, sp, lbl10, prompt, tile, conf, keep_w):
    """-> feats [P,9], labels [P] (DLTB-majority cropland), ws (if keep_w)."""
    z = np.load(Path(dd) / f"{name}.npz")
    rgb = np.ascontiguousarray(z["x6"][0:3].transpose(1, 2, 0))
    Hs, Ws = sp.shape[1:]
    ws = instances_10m(proc, rgb, prompt, tile, Hs, Ws)
    crop = (lbl10 == 1).astype(np.float32); valid = (lbl10 > 0).astype(np.float32)
    feats, labels = [], []
    for w in ws:
        s = w.sum() + 1e-9
        feats.append((sp * w[None]).sum((1, 2)) / s)
        cf = (w * crop).sum() / ((w * valid).sum() + 1e-9)
        labels.append(1 if cf > 0.5 else 0)
    return (np.array(feats).reshape(-1, 9), np.array(labels),
            ws if keep_w else None, z["label"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_obia_rf")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    ex = lambda lst, n: [x for x in lst if (Path(a.data_dir) / f"{x}.npz").exists()][::max(1, len([x for x in lst if (Path(a.data_dir) / f"{x}.npz").exists()]) // n)][:n]
    tr_names = ex(man["train"], a.n_train); te_names = ex(man["test"], a.n_test)
    dicts = [{"county": n.split("_")[0], "idx": int(n.split("_")[1])} for n in tr_names + te_names]
    print(f"[obia-rf] train={len(tr_names)} test={len(te_names)} | prompt='{a.prompt}'", flush=True)

    t0 = time.time()
    cells, _ = parallel_loadsplit_multitemp(dicts, a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=16)
    spec = build_spec(cells); by = {c["name"]: c for c in cells}
    print(f"[obia-rf] 10m spec for {len(spec)} cells ({time.time()-t0:.0f}s)", flush=True)
    proc = load_processor(a.weights, a.device, a.conf)
    print(f"[obia-rf] SAM3 loaded ({time.time()-t0:.0f}s)", flush=True)

    Xtr, ytr = [], []
    for i, n in enumerate(tr_names):
        if n not in spec: continue
        f, l, _, _ = extract_cell(proc, n, a.data_dir, spec[n], by[n]["label"], a.prompt, a.tile, a.conf, False)
        if len(f): Xtr.append(f); ytr.append(l)
        if (i + 1) % 30 == 0: print(f"  train extract {i+1}/{len(tr_names)} ({time.time()-t0:.0f}s)", flush=True)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    print(f"[obia-rf] train parcels={len(ytr)} cropland-frac={ytr.mean():.2f}", flush=True)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1,
                                class_weight="balanced", random_state=0).fit(Xtr, ytr)

    A10 = [0, 0, 0]; A1 = [0, 0, 0]; ZS10 = [0, 0, 0]
    for n in te_names:
        if n not in spec: continue
        f, l, ws, lbl1 = extract_cell(proc, n, a.data_dir, spec[n], by[n]["label"], a.prompt, a.tile, a.conf, True)
        Hs, Ws = spec[n].shape[1:]; H1, W1 = lbl1.shape; lbl10 = by[n]["label"]
        def paint(sel):
            cov = np.zeros((Hs, Ws), np.float32)
            for j in np.where(sel)[0]:
                cov = np.maximum(cov, ws[j])
            return cov >= 0.5
        prob = rf.predict_proba(f)[:, 1] if len(f) else np.array([])
        c10 = paint(prob > 0.5) if len(f) else np.zeros((Hs, Ws), bool)
        zs10 = paint(np.ones(len(ws), bool)) if ws else np.zeros((Hs, Ws), bool)
        for acc, pr10, lb in ((A10, c10, lbl10), (A1, up1m(c10, H1, W1), lbl1),):
            c = f1_counts(pr10, lb); acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]
        c = f1_counts(zs10, lbl10); ZS10[0] += c[0]; ZS10[1] += c[1]; ZS10[2] += c[2]

    rf10 = f1(*A10); rf1 = f1(*A1); zs = f1(*ZS10)
    print(f"\n[obia-rf] === results over {len(te_names)} test cells ===", flush=True)
    print(f"  zero-shot all-parcels: 10m-F1={zs[0]:.4f} (P{zs[1]:.2f}/R{zs[2]:.2f})", flush=True)
    print(f"  SAM3+RF OBIA:          10m-F1={rf10[0]:.4f} (P{rf10[1]:.2f}/R{rf10[2]:.2f}) | 1m-F1={rf1[0]:.4f}", flush=True)
    print(f"  compare: 10m-only pixel baseline 0.853 | UNet-b5 1m 0.838 | SAM3 zero-shot ~0.55", flush=True)
    json.dump({"n_train_cells": len(tr_names), "n_test_cells": len(te_names),
               "train_parcels": int(len(ytr)), "zero_shot_10m_f1": zs[0],
               "rf_10m_f1": rf10[0], "rf_10m_p": rf10[1], "rf_10m_r": rf10[2], "rf_1m_f1": rf1[0]},
              open(out / "obia_rf_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
