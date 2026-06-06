"""Ensemble top-4 best models on 8 balanced test cells.

Models:
  v27  (5ch, B3, F1=0.856)
  v33  (9ch, B3, F1=0.859)
  v34c (9ch, B3 + boundary, F1=0.862)
  v36  (9ch, B5, F1=0.865) ← strongest single
  v37  (17ch, B5 + boundary + 13yr, F1=0.858)

For each test cell:
  - Build inputs per model (5ch/9ch/17ch)
  - Softmax → average probs
  - argmax → F1

Variants:
  (a) v36 alone (baseline best single)
  (b) v27 + v36 avg
  (c) v33 + v34c + v36 avg
  (d) v27 + v33 + v34c + v36 avg
  (e) v27 + v33 + v34c + v36 + v37 avg
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

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3
EXTRA_YEARS_4 = [2018, 2019, 2020, 2022]
EXTRA_YEARS_13 = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                   2018, 2019, 2020, 2022]


def upsample_to(arr2d, H, W):
    t = torch.from_numpy(arr2d.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def load_smp_unet(ckpt, in_channels, backbone, device):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=backbone, encoder_weights=None,
                 in_channels=in_channels, classes=3).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(state)
    return getattr(m, "eval")()


def metrics(prob_3hw, lbl_hw):
    pred = np.argmax(prob_3hw, axis=0)
    v = lbl_hw > 0
    if not v.any(): return None
    pi = (pred == 1) & v; ti = (lbl_hw == 1) & v
    tp = int((pi & ti).sum()); fp = int((pi & ~ti & v).sum())
    fn = int((~pi & ti).sum()); tn = int((~pi & ~ti & v).sum())
    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    iou = tp/(tp+fp+fn) if tp+fp+fn else 0
    return {"f1": f1, "iou": iou, "precision": prec, "recall": rec}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--ndvi-4yr", type=Path, default=HOME / "data/v33_ndvi_multitemporal")
    p.add_argument("--ndvi-13yr", type=Path, default=HOME / "data/v35_ndvi_full")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    device = args.device

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
        nd4_path = args.ndvi_4yr / f"{r['county']}_{r['idx']}.npz"
        nd13_path = args.ndvi_13yr / f"{r['county']}_{r['idx']}.npz"
        if not s2_path.exists(): continue
        s2 = np.load(s2_path)
        rgbnir = s2["rgbnir"].astype(np.float32)
        ndvi_s2 = s2["ndvi"].astype(np.float32)
        Hs, Ws = rgbnir.shape[1], rgbnir.shape[2]
        rgbnir_n = rgbnir.copy()
        for b in range(4): rgbnir_n[b] = (rgbnir_n[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_s2_n = (ndvi_s2 - NDVI_MEAN) / NDVI_STD
        x5 = np.concatenate([rgbnir_n, ndvi_s2_n[None]], 0).astype(np.float32)

        def make_xy(nd_path, year_list):
            if not nd_path.exists(): return None
            nd = np.load(nd_path)
            years = nd["years"].tolist()
            stack = nd["ndvi_years"].astype(np.float32) / 10000.0
            stack_sel = np.stack([stack[years.index(y)] for y in year_list if y in years], 0)
            up = np.stack([upsample_to(stack_sel[i], Hs, Ws)
                            for i in range(len(stack_sel))], 0).astype(np.float32)
            up_n = (up - NDVI_MEAN) / NDVI_STD
            return np.concatenate([rgbnir_n, ndvi_s2_n[None], up_n], 0).astype(np.float32)

        x9 = make_xy(nd4_path, EXTRA_YEARS_4)
        x17 = make_xy(nd13_path, EXTRA_YEARS_13)

        # Rasterize label at S2 grid (WGS84, do NOT reproject)
        bb = tuple(s2["bbox"])
        idx = list(gdf[r["county"]].sindex.intersection(bb))
        label = np.zeros((Hs, Ws), dtype=np.uint8)
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
        cells.append({
            "name": f"{r['county']}_{r['idx']}",
            "x5": x5, "x9": x9, "x17": x17, "label": label,
            "Hs": Hs, "Ws": Ws,
        })
    print(f"  {len(cells)} test cells", flush=True)

    # Load models
    print("[2] load models", flush=True)
    models = {}
    cfg = {
        "v27":  (HOME / "results/v27/best.pt",  5,  "efficientnet-b3"),
        "v33":  (HOME / "results/v33/best.pt",  9,  "efficientnet-b3"),
        "v34c": (HOME / "results/v34c/best.pt", 9,  "efficientnet-b3"),
        "v36":  (HOME / "results/v36/best.pt",  9,  "efficientnet-b5"),
        "v37":  (HOME / "results/v37/best.pt", 17,  "efficientnet-b5"),
        "v35":  (HOME / "results/v35/best.pt", 17,  "efficientnet-b3"),
    }
    for name, (ckpt, ch, bk) in cfg.items():
        if ckpt.exists():
            models[name] = (load_smp_unet(ckpt, ch, bk, device), ch)
            print(f"  loaded {name} ({ch}ch, {bk})", flush=True)
        else:
            print(f"  missing {ckpt}", flush=True)

    # Inference
    print("\n[3] per-cell ensemble", flush=True)
    SZ = 224
    def pad_crop(arr, sz):
        H, W = arr.shape[-2:]
        ph = max(0, sz - H); pw = max(0, sz - W)
        if ph or pw:
            arr = np.pad(arr, ((0,0),(0,ph),(0,pw)), mode="edge")
        return arr[:, :sz, :sz]
    def pad_crop2d(arr, sz):
        H, W = arr.shape
        ph = max(0, sz - H); pw = max(0, sz - W)
        if ph or pw:
            arr = np.pad(arr, ((0,ph),(0,pw)), mode="constant")
        return arr[:sz, :sz]

    results = {k: [] for k in ["v27","v33","v34c","v36","v37","v35",
                                "27_36","36_34c","v36+v37",
                                "27_33_34c_36", "all_avg"]}
    for cell in cells:
        lbl = pad_crop2d(cell["label"], SZ)
        probs = {}
        for name, (m, ch) in models.items():
            x = cell["x5"] if ch == 5 else (cell["x9"] if ch == 9 else cell["x17"])
            if x is None: continue
            xp = pad_crop(x.copy(), SZ)
            with torch.no_grad():
                logits = m(torch.from_numpy(xp)[None].to(device))
                probs[name] = torch.softmax(logits, dim=1)[0].cpu().numpy()
        # Individual
        for k, p in probs.items():
            mt = metrics(p, lbl)
            if mt: results[k].append(mt)
        # Ensembles
        def avg(*names):
            ps = [probs[n] for n in names if n in probs]
            if not ps: return None
            return np.mean(ps, axis=0)
        ensembles = {
            "27_36":         avg("v27","v36"),
            "36_34c":        avg("v36","v34c"),
            "v36+v37":       avg("v36","v37"),
            "27_33_34c_36":  avg("v27","v33","v34c","v36"),
            "all_avg":       avg("v27","v33","v34c","v36","v37","v35"),
        }
        for k, p in ensembles.items():
            if p is None: continue
            mt = metrics(p, lbl)
            if mt: results[k].append(mt)
        line = f"  {cell['name']:>15}: "
        for k in ["v27","v33","v34c","v36","v37","v35","36_34c","27_33_34c_36","all_avg"]:
            if k in probs or k in ensembles:
                last = results[k][-1] if results[k] else None
                line += f"{k}={last['f1']:.3f} " if last else f"{k}=— "
        print(line, flush=True)

    print(f"\n[4] aggregate (avg F1 over {len(cells)} cells)", flush=True)
    print(f"{'Method':<22} {'F1':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}")
    summary = {}
    for k, ms in results.items():
        if not ms: continue
        a = {f: float(np.mean([m[f] for m in ms])) for f in ms[0]}
        print(f"{k:<22} {a['f1']:<8.3f} {a['iou']:<8.3f} {a['precision']:<8.3f} {a['recall']:<8.3f}")
        summary[k] = a
    (HOME / "results/ensemble_top4.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {HOME}/results/ensemble_top4.json")


if __name__ == "__main__":
    main()
