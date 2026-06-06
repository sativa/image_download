"""Evaluate whether CONNECTING fine-tuned DINOv2-1m + fine-tuned SAM3 beats either alone.

For each test cell, compute a 1m cropland mask 4 ways and score F1 vs DLTB:
  1. DINO-alone       : DINOv2-1m per-pixel argmax (the 0.86 baseline, dense, full coverage).
  2. SAM3-FT-alone    : union of all SAM3-FT "crop field" parcels (geometry, recall-limited).
  3. PRODUCT (filter) : SAM3 parcels KEPT only where DINO says cropland (DINO filters SAM3's FPs).
  4. OBIA-refine      : DINO pixels, but each SAM3 parcel snapped to its majority-DINO class
                        (SAM3 boundaries sharpen DINO; pixels outside any parcel keep DINO).
Reports in-domain (Gansu 1m-F1) + cross-province (Changzhi 10m-F1). Answers: does connecting help?
"""
import argparse, json, pickle, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
sys.path.insert(0, "/home/ps/sam3/sam3-inference")
from product import dino_cropland_prob, load_sam3  # reuse DINO prob + SAM3(+ft head) loader
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from torchvision.transforms import v2

DEV = "cuda"


def f1(tp, fp, fn):
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def sam3_masks(proc, rgb, prompt, tile):
    """yield per-instance 1m bool masks (tiled), placed in full-cell coords."""
    H, W = rgb.shape[:2]
    wins = [(y, x, min(y + tile, H), min(x + tile, W)) for y in range(0, H, tile) for x in range(0, W, tile)] if tile else [(0, 0, H, W)]
    for (y0, x0, y1, x1) in wins:
        st = proc.set_image(Image.fromarray(rgb[y0:y1, x0:x1]))
        st = proc.set_text_prompt(prompt=prompt, state=st)
        mk = st.get("masks")
        if mk is None or mk.numel() == 0:
            continue
        for inst in mk.squeeze(1).cpu().numpy():
            full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = inst > 0
            if full.sum() > 200:
                yield full


def eval_cell(dp, masks_iter, lbl, is_10m_lbl=False, gate_thr=0.15):
    """dp = 1m cropland prob; build the 5 masks; return per-method (tp,fp,fn) vs lbl."""
    H, W = dp.shape
    dino = dp >= 0.5
    unc = np.abs(dp - 0.5) < gate_thr   # DINO-uncertain pixels
    sam_u = np.zeros((H, W), bool); prod = np.zeros((H, W), bool); obia = dino.copy(); gated = dino.copy()
    for m in masks_iter:
        sam_u |= m
        crop = dp[m].mean() > 0.5
        if crop:
            prod |= m
        obia[m] = crop                 # snap WHOLE parcel to majority DINO class
        gated[m & unc] = crop          # snap ONLY DINO-uncertain pixels (confidence-gated)
    res = {}
    if is_10m_lbl:  # aggregate each 1m mask to the 10m label grid
        Hs, Ws = lbl.shape
        def to10(a):
            return F.interpolate(torch.from_numpy(a.astype(np.float32))[None, None], size=(Hs, Ws), mode="area")[0, 0].numpy() >= 0.5
        masks = {"dino": to10(dino), "sam3": to10(sam_u), "product": to10(prod), "obia": to10(obia), "gated": to10(gated)}
        v = lbl > 0; gt = (lbl == 1) & v
    else:
        masks = {"dino": dino, "sam3": sam_u, "product": prod, "obia": obia, "gated": gated}
        v = lbl > 0; gt = (lbl == 1) & v
    for k, pm in masks.items():
        pi = pm & v
        res[k] = (int((pi & gt).sum()), int((pi & ~gt & v).sum()), int((~pm & gt).sum()))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dino-ckpt", default="/mnt/sda/zf/landform/results/dino_1m/best.pt")
    p.add_argument("--sam3-weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--sam3-head", default="/mnt/sda/zf/landform/results/sam3_ft_fast/ft_state.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--cz-pkl", default="/mnt/sda/zf/landform/data/changzhi_cells.pkl")
    p.add_argument("--n-cells", type=int, default=40)
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    t0 = time.time()

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    dino = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=4).to(DEV)
    dino.load_state_dict(torch.load(a.dino_ckpt, map_location=DEV, weights_only=True)); dino.eval()
    proc = load_sam3(a.sam3_weights, a.sam3_head)
    print(f"[peval] models loaded ({time.time()-t0:.0f}s)", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    te = te[::max(1, len(te) // a.n_cells)][:a.n_cells]
    AGG = {k: [0, 0, 0] for k in ["dino", "sam3", "product", "obia", "gated"]}
    for i, n in enumerate(te):
        z = np.load(Path(a.data_dir) / f"{n}.npz"); x6 = z["x6"]; lbl = z["label"]
        rgb = np.ascontiguousarray(x6[0:3].transpose(1, 2, 0))
        dp = dino_cropland_prob(dino, x6)
        with torch.no_grad():
            res = eval_cell(dp, sam3_masks(proc, rgb, a.prompt, a.tile), lbl)
        for k in AGG:
            for j in range(3): AGG[k][j] += res[k][j]
        if (i + 1) % 10 == 0: print(f"  in-domain {i+1}/{len(te)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[peval] === IN-DOMAIN (Gansu cross-county, 1m-F1) ===", flush=True)
    out = {}
    for k in ["dino", "sam3", "product", "obia", "gated"]:
        ff, pr, rc = f1(*AGG[k]); out[k] = {"f1": ff, "p": pr, "r": rc}
        print(f"  {k:9s}: F1={ff:.4f} (P{pr:.2f}/R{rc:.2f})", flush=True)

    # cross-province
    cz = pickle.load(open(a.cz_pkl, "rb"))  # trusted internal artifact
    cz_lbl = {c["name"]: c["label"] for c in cz}
    cz_names = [c["name"] for c in cz if (Path(a.cz_1m_dir) / f'{c["name"]}.npz').exists()]
    CZ = {k: [0, 0, 0] for k in AGG}
    for i, n in enumerate(cz_names):
        x6 = np.load(Path(a.cz_1m_dir) / f"{n}.npz")["x6"]; lbl10 = cz_lbl[n]
        rgb = np.ascontiguousarray(x6[0:3].transpose(1, 2, 0))
        dp = dino_cropland_prob(dino, x6)
        with torch.no_grad():
            res = eval_cell(dp, sam3_masks(proc, rgb, a.prompt, a.tile), lbl10, is_10m_lbl=True)
        for k in CZ:
            for j in range(3): CZ[k][j] += res[k][j]
        if (i + 1) % 40 == 0: print(f"  xprov {i+1}/{len(cz_names)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[peval] === CROSS-PROVINCE (Changzhi, 10m-F1) ===", flush=True)
    for k in ["dino", "sam3", "product", "obia", "gated"]:
        ff, pr, rc = f1(*CZ[k]); out[k]["xprov_f1"] = ff
        print(f"  {k:9s}: 10m-F1={ff:.4f} (P{pr:.2f}/R{rc:.2f})", flush=True)
    print(f"\n[peval] does connecting help? compare 'dino'(alone) vs 'product'/'obia'(connected)", flush=True)
    json.dump(out, open("/mnt/sda/zf/landform/results/product/eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
