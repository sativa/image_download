"""v34d: ensemble of v27 + v31 + v33 via averaged softmax probabilities.

All three models output 3-class semantic logits. We resize each to a common
target size (224×224), softmax, average, argmax → final mask.
Evaluated on the same 8 balanced test cells.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS
from train_v33_multitemporal import EXTRA_YEARS

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3


def upsample_to(arr2d, H, W):
    t = torch.from_numpy(arr2d.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def load_v27_or_v33(ckpt, in_channels, device):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                 in_channels=in_channels, classes=3).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(state)
    return getattr(m, "eval")()


def load_v31(ckpt, device):
    from transformers import Mask2FormerForUniversalSegmentation
    from train_v31_mask2former import patch_swin_5ch
    m = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-ade-semantic",
        num_labels=3,
        id2label={0:"nodata",1:"crop",2:"other"},
        label2id={"nodata":0,"crop":1,"other":2},
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    patch_swin_5ch(m, in_channels=5)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m = m.to(device)
    return getattr(m, "eval")()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--ndvi-yr-dir", type=Path, default=HOME / "data/v33_ndvi_multitemporal")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--v27-ckpt", type=Path, default=HOME / "results/v27/best.pt")
    p.add_argument("--v31-ckpt", type=Path, default=HOME / "results/v31/best.pt")
    p.add_argument("--v33-ckpt", type=Path, default=HOME / "results/v33/best.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--out", type=Path, default=HOME / "results/v34d_ensemble.json")
    args = p.parse_args()

    device = args.device
    sz = args.target_size

    # Load test cells
    print("[1] load test cells", flush=True)
    regions = json.loads(args.regions_json.read_text())
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from affine import Affine
    gdf = {}
    for r in regions["test"]:
        c = r["county"]
        if c in gdf: continue
        g = gpd.read_parquet(args.dltb_cache / f"{c}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326: g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf[c] = g

    cells = []
    for r in regions["test"]:
        s2_path = args.s2_dir / f"{r['county']}_{r['idx']}.npz"
        nd_path = args.ndvi_yr_dir / f"{r['county']}_{r['idx']}.npz"
        if not s2_path.exists(): continue
        s2 = np.load(s2_path)
        rgbnir = s2["rgbnir"]; ndvi_s2 = s2["ndvi"]
        Hs, Ws = rgbnir.shape[1], rgbnir.shape[2]
        # Build 5ch input
        rgbnir_n = rgbnir.astype(np.float32).copy()
        for b in range(4): rgbnir_n[b] = (rgbnir_n[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_s2_n = (ndvi_s2.astype(np.float32) - NDVI_MEAN) / NDVI_STD
        x5 = np.concatenate([rgbnir_n, ndvi_s2_n[None]], axis=0).astype(np.float32)
        # 9ch input (for v33)
        if nd_path.exists():
            nd = np.load(nd_path)
            years = nd["years"].tolist()
            stack = nd["ndvi_years"].astype(np.float32) / 10000.0
            stack_sel = np.stack([stack[years.index(y)] for y in EXTRA_YEARS if y in years], 0)
            ndvi_years_up = np.stack([upsample_to(stack_sel[i], Hs, Ws)
                                       for i in range(len(stack_sel))], 0).astype(np.float32)
            ndvi_years_up = (ndvi_years_up - NDVI_MEAN) / NDVI_STD
            x9 = np.concatenate([rgbnir_n, ndvi_s2_n[None], ndvi_years_up], axis=0)
        else:
            x9 = None
        # Rasterize label at S2 grid (WGS84, do NOT reproject to 3857)
        from train_v33_multitemporal import upsample_to as _u  # ensure module loadable
        bb = tuple(s2["bbox"])
        idx = list(gdf[r["county"]].sindex.intersection(bb))
        if idx:
            sub = gdf[r["county"]].iloc[idx].copy()
            sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
            sub = sub[~sub.geometry.is_empty]
            if len(sub) > 0:
                sub["bin"] = np.where((sub["cid"]==1) | (sub["cid"]==2), 1, 2)
                shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["bin"])]
                label = rasterize(shapes=shapes, out_shape=(Hs, Ws),
                                   transform=Affine(*s2["transform"].flatten()[:6]),
                                   fill=0, dtype="uint8")
            else:
                label = np.zeros((Hs, Ws), dtype=np.uint8)
        else:
            label = np.zeros((Hs, Ws), dtype=np.uint8)
        cells.append({
            "name": f"{r['county']}_{r['idx']}",
            "x5": x5, "x9": x9, "label": label, "Hs": Hs, "Ws": Ws,
        })
    print(f"  {len(cells)} test cells", flush=True)

    # Load models
    print("[2] load 3 models", flush=True)
    v27 = load_v27_or_v33(args.v27_ckpt, in_channels=5, device=device)
    v33 = load_v27_or_v33(args.v33_ckpt, in_channels=9, device=device)
    has_v31 = False; v31 = None
    print(f"  skipping v31 (HF model load issues)", flush=True)

    # Inference
    print("[3] inference + ensemble", flush=True)
    from train_v31_mask2former import m2f_to_semantic
    results = {"v27": [], "v31": [], "v33": [], "v27_v33_avg": [], "all_avg": []}
    for cell in cells:
        name = cell["name"]; label = cell["label"]
        Hs, Ws = cell["Hs"], cell["Ws"]

        def pad_to(arr, sz):
            H, W = arr.shape[-2:]
            ph = max(0, sz - H); pw = max(0, sz - W)
            if ph or pw:
                arr = np.pad(arr, ((0,0),(0,ph),(0,pw)), mode="edge")
            return arr[:, :sz, :sz]

        # v27 prob at sz×sz
        x5_p = pad_to(cell["x5"], sz)
        xt = torch.from_numpy(x5_p)[None].to(device)
        with torch.no_grad():
            p27 = torch.softmax(v27(xt), dim=1)[0].cpu().numpy()  # (3, sz, sz)

        # v33 prob
        if cell["x9"] is not None:
            x9_p = pad_to(cell["x9"], sz)
            xt = torch.from_numpy(x9_p)[None].to(device)
            with torch.no_grad():
                p33 = torch.softmax(v33(xt), dim=1)[0].cpu().numpy()
        else:
            p33 = p27

        # v31 prob
        if has_v31:
            xt = torch.from_numpy(x5_p)[None].to(device)
            with torch.no_grad():
                out = v31(pixel_values=xt)
                sem = m2f_to_semantic(out.class_queries_logits,
                                       out.masks_queries_logits,
                                       (sz, sz))
                p31 = sem[0].cpu().numpy()  # already prob-like
                # Normalize to sum 1
                p31 = p31 / p31.sum(axis=0, keepdims=True).clip(1e-6)
        else:
            p31 = p27

        # Pad label too
        lbl_p = label
        if lbl_p.shape != (sz, sz):
            ph = max(0, sz - lbl_p.shape[0]); pw = max(0, sz - lbl_p.shape[1])
            if ph or pw:
                lbl_p = np.pad(lbl_p, ((0,ph),(0,pw)), mode="constant")
            lbl_p = lbl_p[:sz, :sz]

        # Evaluate each + ensemble
        def metrics(prob, lbl):
            pred = np.argmax(prob, axis=0)
            v = lbl > 0
            if not v.any(): return None
            pi = (pred == 1) & v; ti = (lbl == 1) & v
            tp = int((pi & ti).sum()); fp = int((pi & ~ti & v).sum())
            fn = int((~pi & ti).sum()); tn = int((~pi & ~ti & v).sum())
            prec = tp/(tp+fp) if tp+fp else 0
            rec = tp/(tp+fn) if tp+fn else 0
            f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
            iou = tp/(tp+fp+fn) if tp+fp+fn else 0
            return {"f1": f1, "iou": iou, "precision": prec, "recall": rec}

        m27 = metrics(p27, lbl_p)
        m31 = metrics(p31, lbl_p)
        m33 = metrics(p33, lbl_p)
        p2733 = (p27 + p33) / 2
        m_2733 = metrics(p2733, lbl_p)
        p_all = (p27 + p31 + p33) / 3
        m_all = metrics(p_all, lbl_p)
        for k, m in [("v27",m27),("v31",m31),("v33",m33),("v27_v33_avg",m_2733),("all_avg",m_all)]:
            if m: results[k].append(m)
        def f(m): return f"{m['f1']:.3f}" if m else "—"
        print(f"  {name}: v27={f(m27)} v31={f(m31)} v33={f(m33)} "
              f"avg27+33={f(m_2733)} avg_all={f(m_all)}", flush=True)

    print(f"\n[4] aggregate", flush=True)
    summary = {}
    print(f"{'Method':<20} {'F1':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}")
    for k, ms in results.items():
        if not ms: continue
        avg = {f: float(np.mean([m[f] for m in ms])) for f in ms[0]}
        print(f"{k:<20} {avg['f1']:<8.3f} {avg['iou']:<8.3f} {avg['precision']:<8.3f} {avg['recall']:<8.3f}")
        summary[k] = avg
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
