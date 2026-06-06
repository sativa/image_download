"""Supervised land-cover head trained on 三调 (DLTB) ground truth.

End-to-end pipeline:
  1. Sample N training bboxes spread across Heshui County, distinct
     from the held-out test bbox.
  2. Download Esri imagery for each (90 tiles each at z17, parallel).
  3. For each region: SLIC → per-superpixel DINOv2 features + DLTB
     majority label.
  4. Concat all training samples, fit a multinomial logistic regression
     (and optionally an MLP) on (features → DLTB class id).
  5. Apply the trained head to the held-out test region.
  6. Pixel-level evaluation with confusion matrix vs ground truth.

Outputs:
  - `head.joblib` — the trained classifier, loadable by
    `sam3_classify.classify_supervised`.
  - Console summary + confusion matrix.

Why this works where prompts/colour rules fail:
  - Stage 2 now learns from real DLTB polygons that already encode
    "what does each class look like at z17 in Heshui". No more
    cropland-looking-like-bare-soil ambiguity — the network sees the
    same imagery the labels were drawn on.
  - DINOv2 features carry texture + structure information colour
    cannot; an oak forest and an artificial green roof can have the
    same RGB mean but completely different feature signatures.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests


EARTH_HALF_CIRC_M = 20037508.3427892
TILE_PX = 256

DLTB_CLASS_TO_ID = {"耕地": 1, "园地": 2, "林地": 3, "草地": 4, "其他": 5}
ID_TO_DLTB = {v: k for k, v in DLTB_CLASS_TO_ID.items()}


# ──────────── Imagery download (same logic as bench_with_truth.py) ─────────────

def _tile_xy(lon, lat, z):
    n = 2.0 ** z
    lat = max(-85.05112878, min(85.05112878, lat))
    rad = math.radians(lat)
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.log(math.tan(rad) + 1.0/math.cos(rad)) / math.pi) / 2.0 * n))
    return x, y


def _tile_bbox_3857(x, y, z):
    n = 2 ** z
    cell = 2 * EARTH_HALF_CIRC_M / n
    return (-EARTH_HALF_CIRC_M + x * cell, EARTH_HALF_CIRC_M - (y+1)*cell,
            -EARTH_HALF_CIRC_M + (x+1)*cell, EARTH_HALF_CIRC_M - y*cell)


def download_region(bbox_wgs84, zoom, out_tif: Path, session):
    """Fetch all XYZ tiles for one region, stitch, write GeoTIFF."""
    from PIL import Image
    import rasterio
    from rasterio.transform import from_origin
    w, s, e, n = bbox_wgs84
    x_min, y_min = _tile_xy(w, n, zoom)
    x_max, y_max = _tile_xy(e, s, zoom)
    tx, ty = x_max - x_min + 1, y_max - y_min + 1
    canvas = np.zeros((ty * TILE_PX, tx * TILE_PX, 3), dtype=np.uint8)

    def fetch(xy):
        x, y = xy
        for url in [
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}",
            f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}",
        ]:
            try:
                r = session.get(url, timeout=15)
                r.raise_for_status()
                return x, y, r.content
            except Exception:
                continue
        return x, y, None

    tasks = [(x, y) for y in range(y_min, y_max+1) for x in range(x_min, x_max+1)]
    with ThreadPoolExecutor(max_workers=32) as ex:
        for x, y, data in ex.map(fetch, tasks):
            if data:
                img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
                canvas[(y-y_min)*TILE_PX:(y-y_min+1)*TILE_PX,
                       (x-x_min)*TILE_PX:(x-x_min+1)*TILE_PX] = img
    sw, _, _, sn = _tile_bbox_3857(x_min, y_min, zoom)
    se, _, _, _ = _tile_bbox_3857(x_max, y_min, zoom)
    _, ss, _, _ = _tile_bbox_3857(x_min, y_max, zoom)
    H, W = canvas.shape[:2]
    transform = from_origin(sw, sn, (se - sw) / W, (sn - ss) / H)
    profile = {"driver": "GTiff", "height": H, "width": W, "count": 3,
               "dtype": "uint8", "crs": "EPSG:3857", "transform": transform,
               "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(out_tif, "w", **profile) as dst:
        for i in range(3):
            dst.write(canvas[..., i], i+1)
    return out_tif, transform, H, W


def rasterise_dltb(dltb_path, bbox_wgs84, ref_transform, H, W):
    """Clip DLTB to bbox, reproject to EPSG:3857, rasterise on ref grid."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    g = gpd.read_parquet(dltb_path).to_crs("EPSG:4326").clip(shp_box(*bbox_wgs84)).to_crs("EPSG:3857")
    g["cid"] = g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)
    shapes = [(geom, int(cid)) for geom, cid in zip(g.geometry, g["cid"]) if cid > 0]
    if not shapes:
        return np.zeros((H, W), dtype=np.uint8)
    return rasterize(shapes=shapes, out_shape=(H, W), transform=ref_transform, fill=0, dtype="uint8")


# ──────────── Feature extraction ─────────────

class DinoExtractor:
    """DINOv2 feature extractor returning patch-grid features.

    DINOv2's processor resizes inputs to a square (224x224 for base
    DINOv2 configs) and outputs Ph*Pw = (224/14)^2 = 256 patches per
    image. We return BOTH the patch-grid features and the mapping back
    to the original image so the caller can pool per-region.

    Returns:
      (patch_features, (Ph, Pw), (H_orig, W_orig)) where
      patch_features has shape (Ph, Pw, D).
    """

    def __init__(self, device="cpu",
                 weights_dir="/Users/zhangfeng/D/dinov2_weights/dinov2-large"):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(weights_dir)
        self.model = AutoModel.from_pretrained(weights_dir).to(device)
        self.model = getattr(self.model, "eval")()
        self.device = device

    def __call__(self, rgb: np.ndarray):
        import torch
        from PIL import Image
        H, W = rgb.shape[:2]
        pil = Image.fromarray(rgb)
        inp = self.processor(images=pil, return_tensors="pt")
        pv = inp["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=pv)
        tokens = out.last_hidden_state[0, 1:, :]
        _, _, Hi, Wi = pv.shape
        Ph, Pw = Hi // 14, Wi // 14
        feat = tokens.detach().cpu().numpy().reshape(Ph, Pw, -1)
        return feat, (Ph, Pw), (H, W)


def extract_patch_samples(rgb, label_raster, dino_extractor):
    """One sample per DINOv2 patch.

    Each patch's feature comes from DINOv2 directly; its label is the
    majority DLTB class in the corresponding region of the ORIGINAL
    image (patch i,j maps to pixels [i*H/Ph:(i+1)*H/Ph,
    j*W/Pw:(j+1)*W/Pw]). Patches with no labelled pixels are dropped.

    Patch resolution is the inference unit downstream too — so
    train/inference grids match exactly, no interpolation needed.
    """
    feat, (Ph, Pw), (H, W) = dino_extractor(rgb)
    D = feat.shape[-1]
    # Per-patch label = majority DLTB id in the original-image region.
    y_grid = np.zeros((Ph, Pw), dtype=np.int32)
    for i in range(Ph):
        y0 = int(i * H / Ph)
        y1 = int((i + 1) * H / Ph)
        for j in range(Pw):
            x0 = int(j * W / Pw)
            x1 = int((j + 1) * W / Pw)
            region = label_raster[y0:y1, x0:x1]
            labelled = region[region > 0]
            if labelled.size == 0:
                continue
            vals, counts = np.unique(labelled, return_counts=True)
            y_grid[i, j] = int(vals[counts.argmax()])
    X_flat = feat.reshape(-1, D)
    y_flat = y_grid.reshape(-1)
    keep = y_flat > 0
    return X_flat[keep], y_flat[keep], (Ph, Pw), feat, y_grid


# ──────────── Main pipeline ─────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_supervised"))
    p.add_argument("--test-bbox", nargs=4, type=float,
                   default=[107.8631, 35.7523, 107.8831, 35.7723])
    p.add_argument("--train-bboxes", nargs="+",
                   default=[
                       "107.9831,35.7923,108.0031,35.8123",
                       "107.9031,35.8523,107.9231,35.8723",
                       "108.0431,35.6923,108.0631,35.7123",
                       "108.1031,35.8523,108.1231,35.8723",
                   ],
                   help="comma-separated W,S,E,N tuples")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "train/1.0"

    train_bboxes = [tuple(map(float, b.split(","))) for b in args.train_bboxes]
    test_bbox = tuple(args.test_bbox)

    # 1) Download all imagery in parallel ─────────────────────────────
    print(f"\n[1] Downloading {len(train_bboxes)+1} regions in parallel")
    all_bboxes = train_bboxes + [test_bbox]
    paths = [args.out_dir / f"region_{i}.tif" for i in range(len(all_bboxes))]

    def _dl(args_):
        i, bb = args_
        if paths[i].exists():
            import rasterio
            with rasterio.open(paths[i]) as src:
                return i, bb, paths[i], src.transform, src.height, src.width
        return (i, bb) + download_region(bb, args.zoom, paths[i], session)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(all_bboxes)) as ex:
        dl_results = list(ex.map(_dl, list(enumerate(all_bboxes))))
    print(f"  downloads done in {time.time()-t0:.1f}s")

    # 2) Rasterise DLTB for each region ───────────────────────────────
    print(f"\n[2] Rasterising DLTB ground truth")
    region_labels = {}
    for i, bb, tif, transform, H, W in dl_results:
        lbl = rasterise_dltb(args.dltb, bb, transform, H, W)
        cov = (lbl > 0).mean() * 100
        print(f"  region {i}: shape={H}x{W}, coverage={cov:.1f}%")
        region_labels[i] = (tif, transform, H, W, lbl)

    # 3) Extract features (DINOv2 model loaded ONCE, shared across regions) ──
    print(f"\n[3] Extracting DINOv2 patch features per region")
    dino = DinoExtractor()
    train_X, train_y = [], []
    test_data = None  # (rgb, label_raster, transform, feat_grid, y_grid, Ph, Pw)
    import rasterio
    for i, (tif, transform, H, W, lbl) in region_labels.items():
        with rasterio.open(tif) as src:
            bands = src.read(out_dtype="uint8")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        X, y, (Ph, Pw), feat_grid, y_grid = extract_patch_samples(rgb, lbl, dino)
        hist = dict(zip(*np.unique(y, return_counts=True)))
        if i < len(train_bboxes):
            train_X.append(X)
            train_y.append(y)
            print(f"  region {i} (TRAIN): {len(X)} labelled patches "
                  f"(grid {Ph}x{Pw}), classes {hist}")
        else:
            test_data = (rgb, lbl, transform, feat_grid, y_grid, Ph, Pw, H, W)
            print(f"  region {i} (TEST):  {len(X)} labelled patches, "
                  f"classes {hist}")

    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"  TOTAL train: {len(train_X)} superpixels")

    # 4) Train ─────────────────────────────────────────────────────────
    print(f"\n[4] Training LogisticRegression head")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_X)
    Xs = scaler.transform(train_X)
    # scikit-learn ≥ 1.7 dropped the multi_class kwarg; defaults are now
    # multinomial when classes > 2, which is what we want.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs, train_y)
    train_acc = clf.score(Xs, train_y)
    print(f"  train acc: {train_acc:.3f}")

    # Save the head + scaler + class id list.
    import joblib
    model_path = args.out_dir / "head.joblib"
    joblib.dump({"clf": clf, "scaler": scaler, "classes": clf.classes_.tolist(),
                 "id_to_label": ID_TO_DLTB}, model_path)
    print(f"  saved head → {model_path}")

    # 5) Evaluate on test region ──────────────────────────────────────
    print(f"\n[5] Evaluating on held-out test region")
    test_rgb, test_label_raster, _, test_feat_grid, test_y_grid, Ph, Pw, H, W = test_data
    D = test_feat_grid.shape[-1]

    # Predict EVERY patch (including unlabelled ones, for full-image
    # visual coverage) and stamp the class id back onto a pixel raster.
    Xall = test_feat_grid.reshape(-1, D)
    pred_grid = clf.predict(scaler.transform(Xall)).reshape(Ph, Pw)

    # Patch-level evaluation on labelled patches only.
    mask = test_y_grid > 0
    sp_acc = float((pred_grid[mask] == test_y_grid[mask]).mean())
    print(f"  patch-level acc on labelled patches: {sp_acc:.3f}")

    # Stamp prediction onto pixel raster for pixel-level comparison.
    pred_full = np.zeros((H, W), dtype=np.uint8)
    for i in range(Ph):
        y0 = int(i * H / Ph); y1 = int((i + 1) * H / Ph)
        for j in range(Pw):
            x0 = int(j * W / Pw); x1 = int((j + 1) * W / Pw)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    # Pixel-level metrics vs ground-truth raster.
    valid = test_label_raster > 0
    if valid.any():
        p, t = pred_full[valid], test_label_raster[valid]
        acc = float((p == t).mean())
        classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
        classes = [c for c in classes if c != 0]
        ious = {}
        for c in classes:
            inter = int(((p == c) & (t == c)).sum())
            union = int(((p == c) | (t == c)).sum())
            ious[int(c)] = inter / union if union else 0.0
        macro_iou = float(np.mean(list(ious.values())))
        print(f"  pixel-level acc: {acc:.3f}")
        print(f"  per-class IoU:")
        for c in classes:
            print(f"    {ID_TO_DLTB.get(c, c):<6} (id={c}): IoU={ious[c]:.3f}")
        print(f"  macro IoU: {macro_iou:.3f}")

        # Confusion matrix.
        print(f"\n  Confusion (truth → predicted):")
        cls_names = [ID_TO_DLTB.get(c, str(c)) for c in classes]
        print("  " + " " * 8 + " " + " ".join(f"{n:<8}" for n in cls_names))
        for tc, tname in zip(classes, cls_names):
            tm = t == tc
            row = []
            for pc in classes:
                row.append(int(((p == pc) & tm).sum()))
            total = sum(row) or 1
            pct = [f"{r/total*100:>5.1f}%" for r in row]
            print(f"  {tname:<8} " + " ".join(f"{v:<8}" for v in pct))

    # Save the test prediction for visual inspection.
    pred_path = args.out_dir / "test_pred.npy"
    np.save(pred_path, pred_full)
    print(f"\n  saved test prediction array → {pred_path}")


if __name__ == "__main__":
    main()
