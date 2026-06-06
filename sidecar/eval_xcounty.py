"""Cross-county evaluation on the 12 held-out counties (v40 split).

Headline metric = cropland-class (1) F1, computed GLOBALLY (pixels pooled over
all test cells -> robust) at:
  (a) 3-class argmax  (comparable to the training-time number)
  (b) D4 test-time augmentation (8 transforms), if --tta
  (c) leave-one-county-out CV decision threshold (honest, never fit on the
      county it scores) -- the agent-B "tune threshold on a holdout" lever.

Members = the smp models from run_xcounty_ensemble.sh. v39 (DINOv2) is a
separate baseline with its own loader and is evaluated elsewhere.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v33_multitemporal import S2MultiTempDataset, EXTRA_YEARS
from fast_load_multitemp import parallel_loadsplit_multitemp

# (name, arch, encoder, out-dir under results/)
MEMBERS_XC = [
    ("b5unet_s0", "unet",          "efficientnet-b5", "xc_b5unet_s0"),
    ("b5unet_s1", "unet",          "efficientnet-b5", "xc_b5unet_s1"),
    ("b5unet_s2", "unet",          "efficientnet-b5", "xc_b5unet_s2"),
    ("b5unet_s3", "unet",          "efficientnet-b5", "xc_b5unet_s3"),
    ("unetpp",    "unetplusplus",  "efficientnet-b5", "xc_unetpp"),
    ("deeplab",   "deeplabv3plus", "efficientnet-b5", "xc_deeplab"),
    ("segformer", "segformer",     "mit_b5",          "xc_segformer"),
]
# Legacy 9-ch Gansu models (trained on ALL 89 counties). VALID only for the
# Changzhi cross-PROVINCE test (Shanxi unseen) -- NOT for the Gansu cross-county
# test (these models saw the 12 held-out counties).
MEMBERS_LEGACY9 = [
    ("v33",  "unet", "efficientnet-b3", "v33"),
    ("v34c", "unet", "efficientnet-b3", "v34c"),
    ("v36",  "unet", "efficientnet-b5", "v36"),
]


def load_model(arch, enc, ckpt, device):
    import segmentation_models_pytorch as smp
    A = {"unet": smp.Unet, "segformer": smp.Segformer,
         "unetplusplus": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
    m = A(encoder_name=enc, encoder_weights=None, in_channels=9, classes=3).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return getattr(m, "eval")()


def infer(model, x, device, tta):
    """x: (9,H,W) float32 -> softmax (3,H,W). D4 TTA if tta."""
    def run(a):
        a = np.ascontiguousarray(a)
        with torch.no_grad():
            lg = model(torch.from_numpy(a)[None].to(device))
            if lg.shape[-2:] != a.shape[1:]:
                lg = F.interpolate(lg, size=a.shape[1:], mode="bilinear", align_corners=False)
            return torch.softmax(lg, dim=1)[0].cpu().numpy()
    if not tta:
        return run(x)
    acc, n = None, 0
    for k in range(4):
        xr = np.rot90(x, k, axes=(1, 2))
        for fl in (False, True):
            pr = run(xr[:, :, ::-1] if fl else xr)
            if fl: pr = pr[:, :, ::-1]
            pr = np.rot90(pr, -k, axes=(1, 2))
            acc = pr if acc is None else acc + pr
            n += 1
    return acc / n


def _f1(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9), prec, rec


def f1_argmax(probs, labels):
    tp = fp = fn = 0
    for pr, lb in zip(probs, labels):
        v = lb > 0; crop = (pr.argmax(0) == 1) & v; gt = (lb == 1) & v
        tp += int((crop & gt).sum()); fp += int((crop & ~gt & v).sum()); fn += int((~crop & gt).sum())
    return _f1(tp, fp, fn)


def f1_thr(probs, labels, t):
    tp = fp = fn = 0
    for pr, lb in zip(probs, labels):
        v = lb > 0; p1, p2 = pr[1], pr[2]
        crop = (p1 > t * (p1 + p2)) & v; gt = (lb == 1) & v
        tp += int((crop & gt).sum()); fp += int((crop & ~gt & v).sum()); fn += int((~crop & gt).sum())
    return _f1(tp, fp, fn)


def f1_threshold_cv(probs, labels, counties):
    """Leave-one-county-out: threshold chosen on the other counties, applied to held-out.
    Single-region test (e.g. cross-province Changzhi) -> falls back to argmax F1."""
    if len(set(counties)) < 2:
        return f1_argmax(probs, labels)
    ts = np.linspace(0.15, 0.85, 29)
    tp = fp = fn = 0
    for c_out in sorted(set(counties)):
        tr = [i for i, c in enumerate(counties) if c != c_out]
        best_t, best = 0.5, -1.0
        for t in ts:
            f1, _, _ = f1_thr([probs[i] for i in tr], [labels[i] for i in tr], t)
            if f1 > best: best, best_t = f1, t
        for i in (i for i, c in enumerate(counties) if c == c_out):
            v = labels[i] > 0; p1, p2 = probs[i][1], probs[i][2]
            crop = (p1 > best_t * (p1 + p2)) & v; gt = (labels[i] == 1) & v
            tp += int((crop & gt).sum()); fp += int((crop & ~gt & v).sum()); fn += int((~crop & gt).sum())
    return _f1(tp, fp, fn)


def best_global_threshold(probs, labels):
    """Single F1-optimal cropland threshold over the whole set (to derive a transfer threshold)."""
    best = (0.5,) + f1_thr(probs, labels, 0.5)
    for t in np.linspace(0.10, 0.85, 31):
        cur = (float(t),) + f1_thr(probs, labels, float(t))
        if cur[1] > best[1]:
            best = cur
    return best  # (t, f1, prec, rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v40_xcounty_regions.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results", type=Path, default=HOME / "results")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--cells-pkl", type=Path, default=None,
                   help="load test cells from a pickle (e.g. Changzhi cross-province) "
                        "instead of the Gansu v40 split")
    p.add_argument("--member-set", default="xc", choices=["xc", "legacy9"],
                   help="xc = new cross-county ensemble; legacy9 = existing all-89 "
                        "9-ch Gansu models (only valid for the Changzhi cross-province test)")
    p.add_argument("--fixed-threshold", type=float, default=None,
                   help="report ensemble F1 at this cropland-prob threshold (e.g. a "
                        "Gansu-derived threshold transferred to the Changzhi test)")
    p.add_argument("--tag", default="", help="suffix for the output json filename")
    args = p.parse_args()
    members = MEMBERS_XC if args.member_set == "xc" else MEMBERS_LEGACY9

    if args.cells_pkl:
        # Trusted, self-generated bundle (build_changzhi_cells.py, this session, on our
        # own servers) -- not external input, so pickle.load here is safe.
        import pickle
        cells = pickle.loads(args.cells_pkl.read_bytes()); sk = 0
    else:
        regions = json.loads(args.regions_json.read_text())
        cells, sk = parallel_loadsplit_multitemp(
            regions["test"], HOME / "data/v11_dltb", HOME / "data/v19_s2_raw",
            HOME / "data/v33_ndvi_multitemporal", EXTRA_YEARS, max_workers=8)
    ds = S2MultiTempDataset(cells, 224, training=False)
    X = [ds[i][0].numpy() for i in range(len(cells))]
    Y = [ds[i][1].numpy() for i in range(len(cells))]
    counties = [c["name"].split("_")[0] for c in cells]
    print(f"test: {len(cells)} cells / {len(set(counties))} counties (skipped {sk}); TTA={args.tta}", flush=True)

    probs = {}
    for name, arch, enc, od in members:
        ckpt = args.results / od / "best.pt"
        if not ckpt.exists():
            print(f"  [skip] {name}: no checkpoint yet", flush=True); continue
        m = load_model(arch, enc, ckpt, args.device)
        t0 = time.time()
        probs[name] = [infer(m, x, args.device, args.tta) for x in X]
        del m; torch.cuda.empty_cache()
        f1, pr, rc = f1_argmax(probs[name], Y)
        print(f"  {name:11s} F1={f1:.4f} P={pr:.3f} R={rc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    avail = list(probs)
    def ens(names):
        ns = [n for n in names if n in probs]
        return [np.mean([probs[n][i] for n in ns], axis=0) for i in range(len(cells))] if ns else None

    combos = {
        "b5unet_x4": ["b5unet_s0", "b5unet_s1", "b5unet_s2", "b5unet_s3"],
        "diverse":   ["b5unet_s0", "unetpp", "deeplab", "segformer"],
        "all":       avail,
    }
    summary = {}
    for cname, names in combos.items():
        e = ens(names)
        if e is None: continue
        f1a, pra, rca = f1_argmax(e, Y)
        f1c, _, _ = f1_threshold_cv(e, Y, counties)
        bt, bf1, _, _ = best_global_threshold(e, Y)
        rec = {"argmax_f1": round(f1a, 4), "cv_threshold_f1": round(f1c, 4),
               "global_best_t": round(float(bt), 3), "global_best_f1": round(bf1, 4),
               "prec": round(pra, 3), "rec": round(rca, 3),
               "members": [n for n in names if n in probs]}
        msg = f"  ENS[{cname:10s}] argmax={f1a:.4f}  cv_thr={f1c:.4f}  bestT={bt:.2f}->{bf1:.4f}"
        if args.fixed_threshold is not None:
            ff1, _, _ = f1_thr(e, Y, args.fixed_threshold)
            rec["fixed_t"] = args.fixed_threshold
            rec["fixed_t_f1"] = round(ff1, 4)
            msg += f"  fixedT={args.fixed_threshold:.2f}->{ff1:.4f}"
        summary[cname] = rec
        print(msg, flush=True)

    # diagnostic: threshold sweep on the full ensemble (reveals the recall/precision ceiling
    # and whether a low F1 is a calibration problem vs a capability ceiling)
    e_all = ens(avail)
    if e_all is not None:
        sweep = [(round(float(t), 2),) + tuple(round(v, 3) for v in f1_thr(e_all, Y, float(t)))
                 for t in np.linspace(0.10, 0.85, 16)]
        bt = max(sweep, key=lambda x: x[1])
        print("  [diag] ensemble (thr, F1, P, R):", sweep, flush=True)
        print(f"  [diag] oracle-best threshold t={bt[0]} -> F1={bt[1]} (OPTIMISTIC: fit on this test)", flush=True)

    out = {"n_cells": len(cells), "n_counties": len(set(counties)), "tta": args.tta,
           "single": {n: dict(zip(("f1", "prec", "rec"), f1_argmax(probs[n], Y))) for n in avail},
           "ensembles": summary}
    tag = (args.tag + "_") if args.tag else ""
    fn = f"xc_eval_{tag}{'tta' if args.tta else 'argmax'}.json"
    (args.results / fn).write_text(json.dumps(out, indent=2))
    print("wrote", args.results / fn, flush=True)


if __name__ == "__main__":
    main()
