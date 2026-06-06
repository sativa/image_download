"""Cross-province (Changzhi, Shanxi) cropland eval at the 10m-label grid — the only cross-province
ground truth available (no Shanxi DLTB vector). Used to measure whether semi-supervised domain
adaptation on unlabeled Changzhi 1m tiles (semisup_xprov) beats the base model.

For each Changzhi cell: run the (plain 6ch) model on the 1m imagery -> cropland prob -> area-resample
to the 10m label grid -> F1. Reports argmax(0.5) F1 (deployment) and best-threshold F1 (upper bound),
per the CLAUDE.md cross-province reporting convention.
"""
import argparse, pickle, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2


@torch.no_grad()
def prob1m(model, x6, dev, cs=448):
    _, SZ, SZw = x6.shape
    acc = np.zeros((SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xb = torch.from_numpy(norm6(x6[:, t:t + cs, l:l + cs])).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb)
                lg = lg[0] if isinstance(lg, tuple) else lg
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0, 1].cpu().numpy()
            acc[t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
    return acc / np.maximum(cnt, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--cz-pkl", default="/mnt/sda/zf/landform/data/changzhi_cells.pkl")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device

    from transformers import AutoModel
    d = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(d, num_classes=3, in_channels=6, unfreeze_last_n=4).to(dev)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=True)
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()
                           and model.state_dict()[k].shape == v.shape})
    model.eval()

    cz = pickle.load(open(a.cz_pkl, "rb"))  # trusted internal artifact
    cz_lbl = {c["name"]: c["label"] for c in cz}
    names = [c["name"] for c in cz if (Path(a.cz_1m_dir) / f'{c["name"]}.npz').exists()]
    # accumulate confusion at the 10m grid, plus prob/gt arrays for best-thr sweep
    tp = fp = fn = 0
    all_p = []; all_g = []; all_v = []
    for n in names:
        x6 = np.load(Path(a.cz_1m_dir) / f"{n}.npz")["x6"]; lbl = cz_lbl[n]
        Hs, Ws = lbl.shape
        prob = prob1m(model, x6, dev)
        p10 = F.interpolate(torch.from_numpy(prob)[None, None], size=(Hs, Ws), mode="area")[0, 0].numpy()
        v = lbl > 0; gt = (lbl == 1) & v
        pred = (p10 >= 0.5) & v
        tp += int((pred & gt).sum()); fp += int((pred & ~gt & v).sum()); fn += int((~pred & gt & v).sum())
        all_p.append(p10[v]); all_g.append(gt[v])
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); f_arg = 2 * pr * rc / (pr + rc + 1e-9)
    P = np.concatenate(all_p); G = np.concatenate(all_g)
    best = 0; bt = 0.5
    for thr in np.arange(0.3, 0.71, 0.05):
        pd = P >= thr
        t = int((pd & G).sum()); f = int((pd & ~G).sum()); n_ = int((~pd & G).sum())
        pp = t / (t + f + 1e-9); rr = t / (t + n_ + 1e-9); ff = 2 * pp * rr / (pp + rr + 1e-9)
        if ff > best: best = ff; bt = thr
    print(f"[xprov] {Path(a.ckpt).parent.name}: {len(names)} cells | "
          f"argmax 10m-F1={f_arg:.4f} (P{pr:.3f}/R{rc:.3f}) | best-thr F1={best:.4f}@{bt:.2f}", flush=True)


if __name__ == "__main__":
    main()
