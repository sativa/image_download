"""Changzhi (Shanxi) CROSS-PROVINCE eval. Apply the Gansu-trained route-c / baseline ensembles
to the 160 Changzhi cells. For each stage-1 source variant (dual 6ch / esri 3ch / google 3ch),
infer the 1m feat (cropland prob + boundary density) aggregated to the 240x240 Changzhi grid,
then run the route-c ensemble (with that feat) vs the baseline ensemble (zero feat).
Reports cross-province cropland F1 -> 1m net gain + dual-vs-single, all out-of-province."""
import sys, json, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform"); LF = Path("/mnt/sda/zf/landform")
sys.path.insert(0, str(HOME / "sidecar"))
import train_c_stage2 as T  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

DEV = "cuda:0"
SRC_CH = {"dual": [0, 1, 2, 3, 4, 5], "esri": [0, 1, 2], "google": [3, 4, 5]}
ARCH = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "unetplusplus": smp.UnetPlusPlus}
THR = np.round(np.arange(0.10, 0.91, 0.05), 2)


def boundary(m, k=3):
    m = m.unsqueeze(1).float(); p = k // 2
    return (F.max_pool2d(m, k, 1, p) + F.max_pool2d(-m, k, 1, p) * -1).squeeze(1)


@torch.no_grad()
def stage1_feat(ckpt, ch, c1m_dir, names, grid=240, tile=512, stride=384):
    m = smp.Unet(encoder_name="efficientnet-b5", encoder_weights=None, in_channels=len(ch), classes=3).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True)); m.eval()
    feat = {}
    for name in names:
        x6 = np.load(Path(c1m_dir) / f"{name}.npz")["x6"][ch].astype(np.float32) / 255.0
        _, H, W = x6.shape; x = torch.from_numpy(x6).unsqueeze(0).to(DEV)
        acc = torch.zeros((1, 3, H, W), device=DEV); cnt = torch.zeros((1, 1, H, W), device=DEV)
        ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
        xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
        if ys[-1] != max(0, H - tile): ys.append(max(0, H - tile))
        if xs[-1] != max(0, W - tile): xs.append(max(0, W - tile))
        for t in ys:
            for l in xs:
                pa = x[:, :, t:t + tile, l:l + tile]; ph, pw = pa.shape[2], pa.shape[3]
                if ph < tile or pw < tile:
                    pa = F.pad(pa, (0, tile - pw, 0, tile - ph), mode="replicate")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg = m(pa).float()
                acc[:, :, t:t + ph, l:l + pw] += torch.softmax(lg[:, :, :ph, :pw], 1)
                cnt[:, :, t:t + ph, l:l + pw] += 1
        sm = acc / cnt.clamp(min=1); prob = sm[:, 1]; hard = sm.argmax(1); bnd = boundary((hard == 1).float())
        fp = F.interpolate(prob.unsqueeze(1), size=(grid, grid), mode="area")[0, 0]
        fb = F.interpolate(bnd.unsqueeze(1), size=(grid, grid), mode="area")[0, 0]
        feat[name] = torch.stack([fp, fb], 0).cpu().numpy().astype(np.float32)
    del m; torch.cuda.empty_cache()
    return feat


def load_ens(dirs):
    ms = []
    for d in dirs:
        nm = Path(d).name
        arch = "deeplabv3plus" if "_dl" in nm else ("unetplusplus" if "_pp" in nm else "unet")
        m = ARCH[arch](encoder_name="efficientnet-b5", encoder_weights=None, in_channels=11, classes=3).to(DEV)
        m.load_state_dict(torch.load(Path(d) / "best.pt", map_location=DEV, weights_only=True)); m.eval()
        ms.append(m)
    return ms


@torch.no_grad()
def ens_prob(models, x11):
    xb = x11.unsqueeze(0).to(DEV); H, W = x11.shape[1:]
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    xb = F.pad(xb, (0, pw, 0, ph), mode="reflect")
    acc = None
    for m in models:
        a = None
        for k in range(4):
            for fl in (False, True):
                xi = torch.rot90(xb, k, (2, 3))
                if fl: xi = torch.flip(xi, (3,))
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pr = torch.softmax(m(xi).float(), 1)
                if fl: pr = torch.flip(pr, (3,))
                a = (pr := torch.rot90(pr, -k, (2, 3))) if a is None else a + pr
        acc = a / 8 if acc is None else acc + a / 8
    return (acc / len(models))[0, 1, :H, :W].cpu().numpy()


def build9(cell):
    rg = cell["rgbnir"].astype(np.float32).copy()
    for b in range(4):
        rg[b] = (rg[b] - T.S2_MEAN[b]) / T.S2_STD[b]
    nd = (cell["ndvi_s2"].astype(np.float32) - T.NDVI_MEAN) / T.NDVI_STD
    ny = (cell["ndvi_years"].astype(np.float32) - T.NDVI_MEAN) / T.NDVI_STD
    return np.concatenate([rg, nd[None], ny], 0)


def f1(tp, fp, fn):
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9)


def eval_ens(models, cells, feats):
    cnt = {float(t): [0, 0, 0] for t in THR}; am = [0, 0, 0]
    for cell in cells:
        s9 = build9(cell); H, W = cell["label"].shape
        f2 = (feats.get(cell["name"]) if feats else None)
        if f2 is None:
            f2 = np.zeros((2, H, W), np.float32)
        x11 = torch.from_numpy(np.concatenate([s9, f2], 0).astype(np.float32))
        prob = ens_prob(models, x11)
        lab = cell["label"]; v = lab > 0; t1 = (lab == 1) & v
        pa = (prob >= 0.5) & v
        am[0] += int((pa & t1).sum()); am[1] += int((pa & ~t1 & v).sum()); am[2] += int((~pa & t1).sum())
        for t in THR:
            pt = (prob >= t) & v; e = cnt[float(t)]
            e[0] += int((pt & t1).sum()); e[1] += int((pt & ~t1 & v).sum()); e[2] += int((~pt & t1).sum())
    return f1(*am), max(f1(*cnt[float(t)]) for t in THR)


def main():
    # changzhi_cells.pkl is our own artifact (build_changzhi_cells.py) — trusted source.
    cells = pickle.load(open(str(LF / "data/changzhi_cells.pkl"), "rb"))
    man = set(json.load(open(str(LF / "data/c_1m_changzhi/manifest.json")))["cells"])
    cells = [c for c in cells if c["name"] in man]
    names = [c["name"] for c in cells]
    print(f"[changzhi-eval] {len(cells)} cells with 1m", flush=True)

    def present(dirs):
        return [str(LF / "results" / d) for d in dirs if (LF / "results" / d / "best.pt").exists()]
    rc = load_ens(present(["c_stage2_1m", "ens_rc_s1", "ens_rc_s2", "ens_rc_dl", "ens_rc_pp"]))
    bl = load_ens(present(["c_stage2_base", "ens_bl_s1", "ens_bl_s2", "ens_bl_dl", "ens_bl_pp"]))
    bfa, bfg = eval_ens(bl, cells, None)
    print(f"baseline 10m-only: argmax={bfa:.4f} best-thr={bfg:.4f}", flush=True)

    res = {}
    for v, src in [("dual", "dual"), ("esri", "esri"), ("google", "google")]:
        ckpt = LF / "results" / (f"c_stage1{'' if v == 'dual' else '_' + v}") / "best.pt"
        if not ckpt.exists():
            print(f"  route-c [{v}]: stage-1 ckpt missing, skip", flush=True); continue
        feats = stage1_feat(str(ckpt), SRC_CH[src], str(LF / "data/c_1m_changzhi"), names)
        fa, fg = eval_ens(rc, cells, feats)
        res[v] = (fa, fg)
        print(f"route-c [{v}]: argmax={fa:.4f} best-thr={fg:.4f}", flush=True)

    print("\n=== 长治跨省 (cropland F1, 160 cells) ===", flush=True)
    print(f"baseline 10m-only:        argmax={bfa:.4f}  best-thr={bfg:.4f}", flush=True)
    for v in res:
        print(f"route-c (stage-1={v:6}): argmax={res[v][0]:.4f}  best-thr={res[v][1]:.4f}  "
              f"(1m净增益 argmax {res[v][0]-bfa:+.4f})", flush=True)
    json.dump({"baseline": [bfa, bfg], **{v: list(res[v]) for v in res}},
              open(str(LF / "results/changzhi_eval.json"), "w"))


if __name__ == "__main__":
    main()
