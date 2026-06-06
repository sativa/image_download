"""DINO chained multi-expert (per the user's architecture, NOT 15ch early-fusion):

  Expert 1 (DINOv2-1m, geometry+type from 1m RGB): per-pixel cropland probability. CONFIDENT pixels
    keep the DINO decision (domain-invariant, holds cross-province).
  Expert 2 (10m spectral RF, 9-ch): ONLY resolves the UNCERTAIN pixels (|p-0.5|<delta) that DINO can't
    confirm. New, orthogonal spectral info enters only where needed.

Tests whether the chain beats DINOv2-1m alone (in-domain 0.86, cross-province 0.843). delta=0 == pure
DINO. The chain should preserve cross-province (most pixels decided by 1m) while spectral fixes ambiguous
in-domain ones. Reports in-domain 1m-F1 (Gansu) + cross-province 10m-F1 (Changzhi) per delta.
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_route_a import build_spec, EXTRA_YEARS
from fast_load_multitemp import parallel_loadsplit_multitemp
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2


@torch.no_grad()
def dino_prob(model, x6, dev, cs=448):
    """-> (SZ,SZw) cropland probability at 1m."""
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
    acc /= np.maximum(cnt, 1)
    return acc[1] / np.maximum(acc[1] + acc[2], 1e-6)  # P(crop | crop or other), ignore class 0


def up1m(a10, H, W):
    return F.interpolate(torch.from_numpy(a10.astype(np.float32))[None, None], size=(H, W), mode="nearest")[0, 0].numpy()


def f1(tp, fp, fn):
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m/best.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--cz-pkl", default="/mnt/sda/zf/landform/data/changzhi_cells.pkl")
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-yr-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--n-rf", type=int, default=150)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    t0 = time.time()

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    g = lambda ns: [{"county": n.split("_")[0], "idx": int(n.split("_")[1])} for n in ns]
    rf_names = man["train"][::max(1, len(man["train"]) // a.n_rf)][:a.n_rf]
    te_names = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    rfc, _ = parallel_loadsplit_multitemp(g(rf_names), a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=8)
    tec, _ = parallel_loadsplit_multitemp(g(te_names), a.dltb, a.s2_dir, a.ndvi_yr_dir, EXTRA_YEARS, max_workers=8)
    rf_spec = build_spec(rfc); te_spec = build_spec(tec); te_lbl10 = {c["name"]: c["label"] for c in tec}
    print(f"[chain] 10m spec: rf={len(rf_spec)} test={len(te_spec)} ({time.time()-t0:.0f}s)", flush=True)

    # spectral expert: 9-ch pixel RF on Gansu
    X, y = [], []
    for c in rfc:
        sp = rf_spec[c["name"]]; lb = c["label"]; v = lb > 0
        if v.sum() == 0: continue
        idx = np.where(v.ravel())[0]; idx = np.random.choice(idx, min(300, len(idx)), replace=False)
        X.append(sp.reshape(9, -1).T[idx]); y.append((lb.ravel()[idx] == 1).astype(int))
    X = np.concatenate(X); y = np.concatenate(y)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=0).fit(X, y)
    print(f"[chain] spectral RF on {len(y)} px (crop {y.mean():.2f}) ({time.time()-t0:.0f}s)", flush=True)

    # Load DINOv2 (CUDA) AFTER all ProcessPool data loading — CUDA-init-before-fork deadlocks the loaders.
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=4).to(a.device)
    model.load_state_dict(torch.load(a.ckpt, map_location=a.device, weights_only=True)); model.eval()
    print(f"[chain] DINOv2-1m loaded ({time.time()-t0:.0f}s)", flush=True)

    deltas = [0.0, 0.1, 0.2, 0.3, 0.4]

    # ---- in-domain (Gansu), 1m-F1 ----
    agg = {d: [0, 0, 0] for d in deltas}
    for name in te_names:
        if name not in te_spec: continue
        z = np.load(Path(a.data_dir) / f"{name}.npz"); x6 = z["x6"]; lbl = z["label"]; H, W = lbl.shape
        dp = dino_prob(model, x6, a.device)            # 1m cropland prob
        sp = te_spec[name]; Hs, Ws = sp.shape[1:]
        spec10 = rf.predict(sp.reshape(9, -1).T).reshape(Hs, Ws)
        spec1 = up1m(spec10, H, W) > 0.5
        v = lbl > 0; gt = (lbl == 1) & v
        for d in deltas:
            unc = np.abs(dp - 0.5) < d
            final = np.where(unc, spec1, dp >= 0.5) & v
            agg[d][0] += int((final & gt).sum()); agg[d][1] += int((final & ~gt & v).sum()); agg[d][2] += int((~final & gt).sum())
    print(f"\n[chain] === IN-DOMAIN (Gansu cross-county, 1m-F1) ===", flush=True)
    for d in deltas:
        f, pr, rc = f1(*agg[d]); tag = " (=DINOv2 alone)" if d == 0 else ""
        print(f"  delta={d:.1f}: 1m-F1={f:.4f} (P{pr:.2f}/R{rc:.2f}){tag}", flush=True)

    # ---- cross-province (Changzhi), 10m-F1 ----
    cz = pickle.load(open(a.cz_pkl, "rb"))  # trusted internal artifact
    cz_spec = build_spec(cz); cz_lbl = {c["name"]: c["label"] for c in cz}
    cz_names = [c["name"] for c in cz if (Path(a.cz_1m_dir) / f'{c["name"]}.npz').exists()]
    caggr = {d: [0, 0, 0] for d in deltas}
    for name in cz_names:
        x6 = np.load(Path(a.cz_1m_dir) / f"{name}.npz")["x6"]; lbl10 = cz_lbl[name]; Hs, Ws = lbl10.shape
        dp = dino_prob(model, x6, a.device); H, W = dp.shape
        sp = cz_spec[name]; spec10 = rf.predict(sp.reshape(9, -1).T).reshape(Hs, Ws)
        spec1 = up1m(spec10, H, W) > 0.5
        for d in deltas:
            unc = np.abs(dp - 0.5) < d
            final1 = np.where(unc, spec1, dp >= 0.5)
            crop10 = F.interpolate(torch.from_numpy(final1.astype(np.float32))[None, None], size=(Hs, Ws), mode="area")[0, 0].numpy() >= 0.5
            v = lbl10 > 0; gt = (lbl10 == 1) & v; pi = crop10 & v
            caggr[d][0] += int((pi & gt).sum()); caggr[d][1] += int((pi & ~gt & v).sum()); caggr[d][2] += int((~crop10 & gt).sum())
    print(f"\n[chain] === CROSS-PROVINCE (Changzhi, 10m-F1) ===", flush=True)
    res = {}
    for d in deltas:
        f, pr, rc = f1(*caggr[d]); tag = " (=DINOv2 alone)" if d == 0 else ""
        fi, _, _ = f1(*agg[d]); res[str(d)] = {"indomain_1m": fi, "xprov_10m": f}
        print(f"  delta={d:.1f}: 10m-F1={f:.4f} (P{pr:.2f}/R{rc:.2f}){tag}", flush=True)
    print(f"\n[chain] compare: DINOv2-1m alone in-domain 0.86 / cross-province 0.843", flush=True)
    json.dump(res, open(Path(a.ckpt).parent / "chain_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
