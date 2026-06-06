"""Small-parcel attack: fuse the BINARY cropland model with the 12-class model at the PARCEL level.

Diagnosis (parcel_eval.py): unfiltered per-parcel cropland-F1 = 0.68, precision 0.59 — small non-crop
parcels embedded in farmland (house/road/pond) get cropland "bled" into them by the binary model.
The 12-class model has explicit non-crop classes (住宅/交通/水域/工矿…), so it can VETO those.

Per DLTB parcel, compute three cropland decisions and sweep over min-parcel-size:
  binary : binary majority cropland               (the 0.68 baseline)
  cls12  : 12-class majority class in {1耕地,2园地}
  AND    : binary cropland AND cls12 in {1,2}      (intersection -> kills small-parcel false positives)
Reports count-F1 / area-F1 per rule per size threshold. Goal: lift unfiltered count-F1 toward 0.9.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m_v2 import norm6, load_ndvi_full, DinoUNetBoundary
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from parcel_eval import load_county


@torch.no_grad()
def tiled_prob(model, x6, dev, ncls, ndvi=None, cs=448):
    """Tiled softmax-accumulated probability (ncls channels) over a full cell."""
    _, SZ, SZw = x6.shape
    acc = np.zeros((ncls, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xc = norm6(x6[:, t:t + cs, l:l + cs])
            if ndvi is not None:
                xc = np.concatenate([xc, ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb)
                lg = lg[0] if isinstance(lg, tuple) else lg
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
    return acc / np.maximum(cnt, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-ckpt", default="/mnt/sda/zf/landform/results/dino_1m_v2_bnd_mt/snap.pt")
    p.add_argument("--bin-multitemporal", action="store_true", default=True)
    p.add_argument("--cls12-ckpt", default="/mnt/sda/zf/landform/results/dino_12class/best.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--n-cells", type=int, default=120)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device; t0 = time.time()
    bin_ch = 11 if a.bin_multitemporal else 6

    from transformers import AutoModel
    d1 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    binm = DinoUNetBoundary(DinoUNet5ch(d1, num_classes=3, in_channels=bin_ch, unfreeze_last_n=4)).to(dev)
    binm.load_state_dict(torch.load(a.bin_ckpt, map_location=dev, weights_only=True)); binm.eval()
    d2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    clsm = DinoUNet5ch(d2, num_classes=13, in_channels=6, unfreeze_last_n=4).to(dev)
    clsm.load_state_dict(torch.load(a.cls12_ckpt, map_location=dev, weights_only=True)); clsm.eval()
    print(f"[fused] binary={Path(a.bin_ckpt).parent.name}(ch{bin_ch}) + 12class loaded ({time.time()-t0:.0f}s)", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()][:a.n_cells]
    cache = {}
    parcels = []  # (area, true_crop, bin_crop, crop12)
    for i, n in enumerate(te):
        z = np.load(Path(a.data_dir) / f"{n}.npz"); x6 = z["x6"]; bbox = z["bbox"]
        _, H, W = x6.shape
        ndvi = load_ndvi_full(n, H, W) if a.bin_multitemporal else None
        prob_bin = tiled_prob(binm, x6, dev, 3, ndvi=ndvi)[1]            # P(cropland)
        cls12 = tiled_prob(clsm, x6, dev, 13).argmax(0)                  # per-pixel 12-class argmax
        try:
            g = load_county(a.dltb, n.split("_")[0], cache)
        except Exception as ex:
            print(f"  skip {n}: {ex}", flush=True); continue
        tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
        idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
        if not idx:
            continue
        cb = shp_box(*bbox); sub = g.iloc[idx].reset_index(drop=True)
        shapes = [(geom, j + 1) for j, geom in enumerate(sub.geometry) if geom.intersects(cb)]
        if not shapes:
            continue
        pid = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        for j in range(len(sub)):
            m = pid == (j + 1); area = int(m.sum())
            if area < 1:
                continue
            true_crop = int(sub["cid"].iloc[j]) in (1, 2)
            bin_crop = bool(prob_bin[m].mean() >= 0.5)
            vals, cnts = np.unique(cls12[m], return_counts=True)
            maj12 = int(vals[cnts.argmax()])
            crop12 = maj12 in (1, 2)
            parcels.append((area, true_crop, bin_crop, crop12))
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(te)} parcels={len(parcels)} ({time.time()-t0:.0f}s)", flush=True)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9), pr, rc

    rules = {"binary": lambda ar, b, c: b, "cls12": lambda ar, b, c: c, "AND": lambda ar, b, c: b and c,
             "cond2k": lambda ar, b, c: c if ar < 2000 else b,    # small->12class, large->binary
             "cond5k": lambda ar, b, c: c if ar < 5000 else b}
    print(f"\n[fused] === {len(parcels)} parcels; count-F1 (P/R) per rule per min-px ===", flush=True)
    out = {}
    for thr in [1, 500, 1000, 2000, 5000]:
        line = f"  >={thr:<5d}"
        for rn, rf in rules.items():
            tp = fp = fn = 0
            for area, tc, b, c in parcels:
                if area < thr:
                    continue
                pc = rf(area, b, c)
                if pc and tc: tp += 1
                elif pc and not tc: fp += 1
                elif not pc and tc: fn += 1
            ff, pp, rr = f1(tp, fp, fn)
            line += f"  {rn}={ff:.3f}(P{pp:.2f}/R{rr:.2f})"
            out.setdefault(rn, {})[thr] = ff
        print(line, flush=True)
    json.dump(out, open("/mnt/sda/zf/landform/results/parcel_eval_fused.json", "w"), indent=2)


if __name__ == "__main__":
    main()
