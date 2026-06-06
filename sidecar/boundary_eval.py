"""Quantitative boundary-delineation quality of the parcel-boundary head vs DLTB ground-truth edges
(the standard field-delineation metric, à la Waldner & Diakogiannis). For Gansu test cells: predict the
boundary probability, threshold to an edge mask, and compare to the c_1m_pbound DLTB-edge labels with a
pixel tolerance -> boundary Precision / Recall / F1 at tol = 0/2/3 px."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import load_ndvi_full
from train_dino_1m_v3 import DinoV3FreqUNet, DinoV3FreqUNetBD, DINOV3_SAT


@torch.no_grad()
def bnd_prob(model, x6, ndvi, dev, cs=448):
    _, SZ, SZw = x6.shape
    acc = np.zeros((SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t + cs, l:l + cs]), ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, bnd, _ = model(xb)
                if bnd.shape[-2:] != (cs, cs):
                    bnd = F.interpolate(bnd, size=(cs, cs), mode="bilinear", align_corners=False)
                pb = torch.sigmoid(bnd.float())[0, 0].cpu().numpy()
            acc[t:t + cs, l:l + cs] += pb; cnt[t:t + cs, l:l + cs] += 1
    return acc / np.maximum(cnt, 1)


def bf1(pred_b, true_b, tol):
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        tdil = cv2.dilate(true_b, k) > 0; pdil = cv2.dilate(pred_b, k) > 0
    else:
        tdil = true_b > 0; pdil = pred_b > 0
    pb = pred_b > 0; tb = true_b > 0
    prec = (pb & tdil).sum() / max(1, pb.sum())
    rec = (tb & pdil).sum() / max(1, tb.sum())
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1, prec, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_v3_8class_bh/best.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_lc")
    p.add_argument("--pbound-dir", default="/mnt/sda/zf/landform/data/c_1m_pbound")
    p.add_argument("--ncls", type=int, default=9)
    p.add_argument("--n-cells", type=int, default=60)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--boundary-decoder", action="store_true")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device
    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    Net = DinoV3FreqUNetBD if a.boundary_decoder else DinoV3FreqUNet
    model = Net(d3, num_classes=a.ncls, in_channels=11, unfreeze_last_n=4).to(dev)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=True); msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()
    te = [n for n in json.loads((Path(a.data_dir) / "manifest.json").read_text())["test"]
          if (Path(a.pbound_dir) / f"{n}.npy").exists()][:a.n_cells]
    print(f"[bnd-eval] {len(te)} Gansu test cells, thr={a.thr}", flush=True)
    agg = {0: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}   # tol -> [sum_f1,sum_p,sum_r]
    n = 0
    for nm in te:
        x6 = np.load(Path(a.data_dir) / f"{nm}.npz")["x6"]; _, SZ, SZw = x6.shape
        ndvi = load_ndvi_full(nm, SZ, SZw)
        if ndvi is None: ndvi = np.zeros((5, SZ, SZw), np.float32)
        pb = (bnd_prob(model, x6, ndvi, dev) >= a.thr).astype(np.uint8)
        tb = (np.load(Path(a.pbound_dir) / f"{nm}.npy") > 0).astype(np.uint8)
        for tol in (0, 2, 3):
            f1, pr, rc = bf1(pb, tb, tol)
            agg[tol][0] += f1; agg[tol][1] += pr; agg[tol][2] += rc
        n += 1
        if n % 20 == 0: print(f"  {n}/{len(te)}", flush=True)
    print("\n=== 边界 delineation 质量 (边界头 vs DLTB 全图斑边界) ===", flush=True)
    for tol in (0, 2, 3):
        s = agg[tol]
        print(f"  tol={tol}px  boundary-F1={s[0]/n:.4f}  P={s[1]/n:.3f}  R={s[2]/n:.3f}", flush=True)


if __name__ == "__main__":
    main()
