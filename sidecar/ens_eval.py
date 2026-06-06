"""Ensemble eval for route-c: average member probabilities over the 120 cross-county
test cells, then leave-county-out threshold CV F1. Baseline members are fed ZERO
1m-feature channels (they were trained with --no-1m); route-c members get the real
stage-1 features. Reports baseline-ensemble vs route-c-ensemble (the 1m net gain),
vs the 10m@20k ensemble reference (0.853)."""
import sys, json, copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HOME = Path("/home/ps/landform")
LF = Path("/mnt/sda/zf/landform")
sys.path.insert(0, str(HOME / "sidecar"))
import train_c_stage2 as T  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

DEV = "cuda:0"
ARCH = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus,
        "unetplusplus": smp.UnetPlusPlus, "segformer": smp.Segformer}


def load_model(d):
    name = Path(d).name
    arch = "deeplabv3plus" if "_dl" in name else ("unetplusplus" if "_pp" in name else "unet")
    m = ARCH[arch](encoder_name="efficientnet-b5", encoder_weights=None, in_channels=11, classes=3).to(DEV)
    m.load_state_dict(torch.load(Path(d) / "best.pt", map_location=DEV, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def ensemble_eval(members, cells, names):
    loader = torch.utils.data.DataLoader(T.Stage2DS(cells, 224, False), batch_size=16,
                                         shuffle=False, num_workers=4)
    member_probs, ys = [], []
    for k, d in enumerate(members):
        m = load_model(d)
        probs = []
        for x, y in loader:
            probs.append(T.tta_prob(m, x.to(DEV), DEV).cpu())
            if k == 0:
                ys.append(y)
        member_probs.append(torch.cat(probs))
        del m
        torch.cuda.empty_cache()
    ens = torch.stack(member_probs).mean(0)  # (N,H,W)
    Y = torch.cat(ys)
    byc = defaultdict(lambda: {"am": [0, 0, 0], "thr": {float(t): [0, 0, 0] for t in T.THRS}})
    for j in range(ens.shape[0]):
        cty = names[j].split("_")[0]
        v = Y[j] > 0
        t1 = (Y[j] == 1) & v
        d = byc[cty]
        pa = (ens[j] >= 0.5) & v
        d["am"][0] += int((pa & t1).sum()); d["am"][1] += int((pa & ~t1 & v).sum()); d["am"][2] += int((~pa & t1).sum())
        for t in T.THRS:
            pt = (ens[j] >= t) & v
            e = d["thr"][float(t)]
            e[0] += int((pt & t1).sum()); e[1] += int((pt & ~t1 & v).sum()); e[2] += int((~pt & t1).sum())
    A = np.sum([d["am"] for d in byc.values()], 0)
    f1_am = T.f1_from_counts(*A)
    counties = list(byc)
    cv = [0, 0, 0]
    for c in counties:
        sc = {float(t): T.f1_from_counts(*np.sum([byc[o]["thr"][float(t)] for o in counties if o != c], 0)) for t in T.THRS}
        ts = max(sc, key=sc.get)
        e = byc[c]["thr"][ts]
        cv[0] += e[0]; cv[1] += e[1]; cv[2] += e[2]
    return f1_am, T.f1_from_counts(*cv)


def main():
    R = json.loads((HOME / "data/v40_5k.json").read_text())
    te, _ = T.parallel_loadsplit_multitemp(R["test"], HOME / "data/v11_dltb", HOME / "data/v19_s2_raw",
                                           HOME / "data/v33_ndvi_multitemporal", T.EXTRA_YEARS, max_workers=8)
    names = [c["name"] for c in te]
    T.attach_feat(te, str(LF / "data/c_stage1_feat"), True)
    te_real = te
    te_zero = copy.deepcopy(te)
    for c in te_zero:
        c["feat2"] = np.zeros_like(c["feat2"])

    def present(dirs):
        return [str(LF / "results" / d) for d in dirs if (LF / "results" / d / "best.pt").exists()]

    bl = present(["c_stage2_base", "ens_bl_s1", "ens_bl_s2", "ens_bl_dl", "ens_bl_pp"])
    rc = present(["c_stage2_1m", "ens_rc_s1", "ens_rc_s2", "ens_rc_dl", "ens_rc_pp"])
    print(f"baseline members ({len(bl)}): {[Path(x).name for x in bl]}", flush=True)
    print(f"route-c  members ({len(rc)}): {[Path(x).name for x in rc]}", flush=True)
    bl_am, bl_cv = ensemble_eval(bl, te_zero, names)
    rc_am, rc_cv = ensemble_eval(rc, te_real, names)
    print("\n=== ENSEMBLE cross-county F1 (120 held-out cells / 12 counties) ===", flush=True)
    print(f"baseline 10m-only @5k ({len(bl)}-model): argmax={bl_am:.4f}  leave-county-CV={bl_cv:.4f}", flush=True)
    print(f"route-c  10m+1m   @5k ({len(rc)}-model): argmax={rc_am:.4f}  leave-county-CV={rc_cv:.4f}", flush=True)
    print(f"\n>>> 1m NET GAIN (ensemble CV F1): {rc_cv - bl_cv:+.4f}", flush=True)
    print(">>> reference: 10m ensemble @20k-train cross-county F1 = 0.853 / acc 0.906", flush=True)
    (LF / "results/ens_compare.json").write_text(json.dumps(
        {"baseline_cv": bl_cv, "routec_cv": rc_cv, "gain": rc_cv - bl_cv,
         "baseline_argmax": bl_am, "routec_argmax": rc_am,
         "n_baseline": len(bl), "n_routec": len(rc)}, indent=2))


if __name__ == "__main__":
    main()
