"""Cross-province test of the fine-tuned DINOv2-1m model (Gansu-trained -> Changzhi/Shanxi).

Key question: DINOv2-1m uses ONLY 1m RGB (no 10m spectra), so like SAM3 its features are
texture/geometry-based and MIGHT generalize cross-province where spectral models collapse
(trained 10m argmax 0.236). Tiled 1m inference -> area-pool the 1m cropland prediction to the
10m grid -> F1 vs the Changzhi 10m DLTB label (no 1m GT exists for Changzhi).
"""
import argparse, json, pickle, sys, time
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
def predict_1m(model, x6, dev, cs=448):
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
    return (acc / np.maximum(cnt, 1)).argmax(0)  # (SZ,SZw) 1m argmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m/best.pt")
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--cz-pkl", default="/mnt/sda/zf/landform/data/changzhi_cells.pkl")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    t0 = time.time()

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=4).to(a.device)
    model.load_state_dict(torch.load(a.ckpt, map_location=a.device, weights_only=True))
    model.eval()
    print(f"[dino-cz] model loaded ({time.time()-t0:.0f}s)", flush=True)

    # trusted internal artifact (changzhi_fuse.py output): 10m labels for Changzhi
    cz = pickle.load(open(a.cz_pkl, "rb"))
    cz_lbl = {c["name"]: c["label"] for c in cz}
    names = [c["name"] for c in cz if (Path(a.cz_1m_dir) / f'{c["name"]}.npz').exists()]
    print(f"[dino-cz] {len(names)} Changzhi cells", flush=True)

    tp = fp = fn = 0
    for i, n in enumerate(names):
        x6 = np.load(Path(a.cz_1m_dir) / f"{n}.npz")["x6"]; lbl = cz_lbl[n]
        Hs, Ws = lbl.shape
        pred1m = predict_1m(model, x6, a.device, a.crop)
        cropf = F.interpolate(torch.from_numpy((pred1m == 1).astype(np.float32))[None, None],
                              size=(Hs, Ws), mode="area")[0, 0].numpy()
        p10 = cropf >= 0.5; v = lbl > 0; gt = (lbl == 1) & v; pi = p10 & v
        tp += int((pi & gt).sum()); fp += int((pi & ~gt & v).sum()); fn += int((~p10 & gt).sum())
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); f1 = 2 * pr * rc / (pr + rc + 1e-9)
    print(f"\n[dino-cz] === DINOv2-1m CROSS-PROVINCE (Gansu->Changzhi, {len(names)} cells) ===", flush=True)
    print(f"  10m-aggregated F1={f1:.4f} (P{pr:.2f}/R{rc:.2f})", flush=True)
    print(f"  compare: trained 10m spectral argmax 0.236 | SAM3+OBIA 0.649 | DINOv2-1m in-domain 1m-F1 0.860", flush=True)
    json.dump({"n_cells": len(names), "xprov_10m_f1": f1, "p": pr, "r": rc},
              open(Path(a.ckpt).parent / "changzhi_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
