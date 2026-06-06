"""v6: Esri + Google multi-source data augmentation + DINOv2 MLP head.

Same 12-region grid + 5-class DLTB scheme as v3, but each region is
downloaded TWICE — once from Esri World Imagery, once from Google
Satellite. The two providers use different capture dates, sensors and
colour pipelines, so the same patch of ground gets two visually
distinct training samples sharing one ground-truth label. Free 2×
augmentation that catches model brittleness to colour balance / season.

The downloader (modified from train_supervised) now takes a source
arg. Caching key includes source so each tile is fetched once.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests


sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_supervised import (
    DLTB_CLASS_TO_ID, ID_TO_DLTB, _tile_xy, _tile_bbox_3857,
)
from train_v2 import TRAIN_BBOXES, TEST_BBOX
from train_v3 import DinoExtractorHiRes, extract_patch_samples


TILE_PX = 256


def download_region_source(bbox_wgs84, zoom, source: str, out_tif: Path, session):
    """Same logic as train_supervised.download_region but with explicit
    source selection. `source` in {"esri", "google"}."""
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
        if source == "esri":
            url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                   f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}")
        elif source == "google":
            url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
        else:
            raise ValueError(f"unknown source {source}")
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            return x, y, r.content
        except Exception:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v6"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--sources", default="esri,google",
                   help="comma-separated tile sources to include in training")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = args.sources.split(",")
    session = requests.Session(); session.headers["User-Agent"] = "trainv6/1.0"

    # 1) Download all (region × source) combinations in parallel ──────
    # Train regions are downloaded TWICE (one per source); the test
    # region is downloaded only once from Esri (matches v3 evaluation
    # so the head-to-head is direct).
    all_jobs = []  # (key, bbox, source, path)
    for i, bb in enumerate(TRAIN_BBOXES):
        for src in sources:
            all_jobs.append((f"train_{i}_{src}", bb, src,
                             args.out_dir / f"train_{i}_{src}.tif"))
    all_jobs.append(("test", TEST_BBOX, "esri",
                     args.out_dir / "test_esri.tif"))

    print(f"[1] Downloading {len(all_jobs)} (region × source) tiles in parallel")
    def _dl(job):
        key, bb, src, path = job
        if path.exists():
            import rasterio
            with rasterio.open(path) as rs:
                return key, bb, src, path, rs.transform, rs.height, rs.width
        out = download_region_source(bb, args.zoom, src, path, session)
        return (key, bb, src) + out

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(all_jobs), 32)) as ex:
        dl_results = list(ex.map(_dl, all_jobs))
    print(f"  done in {time.time()-t0:.1f}s")

    # 2) Rasterise DLTB (one label raster per UNIQUE region) ──────────
    print(f"\n[2] Rasterising DLTB once per unique region")
    import geopandas as gpd
    from shapely.geometry import box as shp_box
    from rasterio.features import rasterize as _rasterize
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try:
        full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError:
        full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)

    # Group jobs by bbox+transform so we don't rasterise twice for the
    # same region's two source versions (transforms are identical when
    # zoom/bbox match, modulo tile-snap which is identical too).
    label_cache: dict = {}
    def _label_for(bb, transform, H, W):
        key = (bb, H, W)
        if key in label_cache:
            return label_cache[key]
        idx = list(full_g.sindex.intersection(bb))
        sub = full_g.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
        sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
        shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
        lbl = (_rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                          fill=0, dtype="uint8")
               if shapes else np.zeros((H, W), dtype=np.uint8))
        label_cache[key] = lbl
        return lbl

    # 3) Extract DINOv2 features for every (region × source) ───────────
    print(f"\n[3] Extracting DINOv2 features @ {args.image_size}px")
    dino = DinoExtractorHiRes(image_size=args.image_size)
    import rasterio
    train_X, train_y = [], []
    test_data = None
    t0 = time.time()
    for key, bb, src, path, transform, H, W in dl_results:
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        lbl = _label_for(bb, transform, H, W)
        X, y, (Ph, Pw), feat_grid, y_grid = extract_patch_samples(rgb, lbl, dino)
        if key == "test":
            test_data = (rgb, lbl, feat_grid, y_grid, Ph, Pw, H, W)
        else:
            train_X.append(X); train_y.append(y)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"  TOTAL train: {len(train_X)} patches in {time.time()-t0:.1f}s")
    print(f"  patches/region avg: {len(train_X)/len(TRAIN_BBOXES)/len(sources):.0f}")
    print(f"  class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    # 4) Train MLP head ─────────────────────────────────────────────────
    print(f"\n[4] Training MLP")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_X)
    Xs = scaler.transform(train_X)
    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                        early_stopping=True, random_state=0)
    clf.fit(Xs, train_y)
    print(f"  trained in {time.time()-t0:.1f}s, train_acc={clf.score(Xs, train_y):.3f}")

    import joblib
    joblib.dump({"clf": clf, "scaler": scaler, "image_size": args.image_size,
                 "id_to_label": ID_TO_DLTB, "sources": sources},
                args.out_dir / "head_v6.joblib")

    # 5) Evaluate ───────────────────────────────────────────────────────
    print(f"\n[5] Evaluation")
    rgb_test, lbl_test, feat_grid_test, y_grid_test, Ph, Pw, H, W = test_data
    D = feat_grid_test.shape[-1]
    pred_grid = clf.predict(scaler.transform(feat_grid_test.reshape(-1, D))).reshape(Ph, Pw)
    pred_full = np.zeros((H, W), dtype=np.uint8)
    for i in range(Ph):
        y0, y1 = int(i*H/Ph), int((i+1)*H/Ph)
        for j in range(Pw):
            x0, x1 = int(j*W/Pw), int((j+1)*W/Pw)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = lbl_test > 0
    p, t = pred_full[valid], lbl_test[valid]
    acc = float((p == t).mean())
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(ious.values())))

    print(f"  test acc: {acc:.3f}")
    for c in classes:
        print(f"    {ID_TO_DLTB.get(c, str(c)):<6}: IoU={ious[c]:.3f}")
    print(f"  macro IoU: {macro:.3f}")
    print()
    print(f"  vs v3 (Esri only):  acc 0.383  macro_IoU 0.238")
    print(f"  v6 (Esri+Google):   acc {acc:.3f}  macro_IoU {macro:.3f}")


if __name__ == "__main__":
    main()
