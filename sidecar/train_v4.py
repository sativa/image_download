"""v4: 30 training regions + 8-class scheme (split "其他" by DLBM).

Two big upgrades over v3:

  1. **More data**:  30 spatially-diverse 2-km regions (k-means cluster
     the 551 candidate cells' centroids, pick 30 representatives).

  2. **Finer class scheme**:  Use the DLTB `DLBM` (国标 GB/T 21010)
     two-digit prefix to break "其他" into homogeneous sub-classes:
       * 05–09 → 建设  (residential / industrial / public)
       * 10    → 交通  (transportation)
       * 11    → 水域  (water)
       * 12    → 裸地  (bare / misc)
     Keeping the 4 primary classes from v3 unchanged, total = 8.

Everything else (DINOv2 448, MLP 256+128, parallel everything) stays
the same so the only variables are data and class taxonomy.
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
from train_supervised import download_region
from train_v3 import DinoExtractorHiRes, extract_patch_samples


# 8-class scheme: 4 primary from v3 + 4 sub-classes from former "其他".
CLASS_TO_ID = {
    "耕地": 1, "园地": 2, "林地": 3, "草地": 4,
    "建设": 5, "交通": 6, "水域": 7, "裸地": 8,
}
ID_TO_LABEL = {v: k for k, v in CLASS_TO_ID.items()}


def dlbm_to_class_id(dlbm) -> int:
    """Map a DLTB DLBM code (string) to our 8-class id. Returns 0 on miss."""
    if dlbm is None or (isinstance(dlbm, float) and math.isnan(dlbm)):
        return 0
    s = str(dlbm).strip()
    if not s:
        return 0
    # First 2 digits are the 一级地类 code (国标 GB/T 21010).
    head = s[:2]
    if head == "01":
        return 1  # 耕地
    if head == "02":
        return 2  # 园地
    if head == "03":
        return 3  # 林地
    if head == "04":
        return 4  # 草地
    if head in ("05", "06", "07", "08", "09"):
        return 5  # 建设
    if head == "10":
        return 6  # 交通
    if head == "11":
        return 7  # 水域
    if head == "12":
        return 8  # 裸地
    return 0


def pick_diverse_train_regions(g_wgs84, n_regions: int, step: float = 0.02) -> list[tuple[float, float, float, float]]:
    """Sample N spatially-diverse 2-km cells, each with ≥4 一级地类 classes."""
    W, S, E, N = g_wgs84.total_bounds
    candidates = []
    for ny in np.arange(S, N - step, step):
        for nx in np.arange(W, E - step, step):
            bb = (nx, ny, nx + step, ny + step)
            idx = list(g_wgs84.sindex.intersection(bb))
            if not idx:
                continue
            sub = g_wgs84.iloc[idx]
            if sub["一级地类"].nunique() >= 4:
                candidates.append((nx + step / 2, ny + step / 2, nx, ny))  # (cx, cy, w, s)
    if not candidates:
        raise RuntimeError("no candidate cells found")
    pts = np.array([(c[0], c[1]) for c in candidates], dtype=np.float64)
    # K-means cluster centroids → pick n_regions points farthest from
    # each other for spatial diversity.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(n_regions, len(candidates)), random_state=0, n_init=10).fit(pts)
    # Pick the candidate closest to each cluster centroid.
    chosen = set()
    bboxes = []
    for center in km.cluster_centers_:
        dists = np.linalg.norm(pts - center, axis=1)
        order = np.argsort(dists)
        for k in order:
            if k not in chosen:
                chosen.add(k)
                cx, cy, w, s = candidates[k]
                bboxes.append((w, s, w + step, s + step))
                break
    return bboxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v4"))
    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--test-bbox", nargs=4, type=float,
                   default=[107.8631, 35.7523, 107.8831, 35.7723])
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers["User-Agent"] = "trainv4/1.0"

    # 1) Pick training regions ──────────────────────────────────────────
    print(f"[1] Picking {args.n_train} spatially-diverse training regions")
    import geopandas as gpd
    t0 = time.time()
    g_w = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try:
        g_w["geometry"] = g_w.geometry.make_valid()
    except AttributeError:
        g_w["geometry"] = g_w.geometry.buffer(0)
    g_w["cid"] = g_w["DLBM"].map(dlbm_to_class_id).fillna(0).astype(int)
    print(f"  loaded + fixed {len(g_w)} polygons in {time.time()-t0:.1f}s")
    print(f"  8-class hist: {dict(zip(*np.unique(g_w['cid'].values, return_counts=True)))}")

    # Exclude the test bbox from candidates so no spatial overlap.
    train_bboxes = pick_diverse_train_regions(g_w, args.n_train)
    test_bbox = tuple(args.test_bbox)
    # Drop any candidate whose centre falls inside the test bbox.
    train_bboxes = [bb for bb in train_bboxes if not (
        test_bbox[0] <= (bb[0]+bb[2])/2 <= test_bbox[2] and
        test_bbox[1] <= (bb[1]+bb[3])/2 <= test_bbox[3]
    )]
    print(f"  selected {len(train_bboxes)} training bboxes (test held out)")

    # 2) Download all regions in parallel ───────────────────────────────
    all_bboxes = train_bboxes + [test_bbox]
    paths = [args.out_dir / f"region_{i}.tif" for i in range(len(all_bboxes))]
    print(f"\n[2] Downloading {len(all_bboxes)} regions in parallel")
    def _dl(args_):
        i, bb = args_
        if paths[i].exists():
            import rasterio
            with rasterio.open(paths[i]) as src:
                return i, bb, paths[i], src.transform, src.height, src.width
        return (i, bb) + download_region(bb, args.zoom, paths[i], session)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(all_bboxes), 32)) as ex:
        dl_results = list(ex.map(_dl, list(enumerate(all_bboxes))))
    print(f"  done in {time.time()-t0:.1f}s")

    # 3) Rasterise DLTB per region (parallel) ───────────────────────────
    print(f"\n[3] Rasterising DLTB per region (8-class)")
    from shapely.geometry import box as shp_box
    from rasterio.features import rasterize as _rasterize
    region_labels = {}

    def _ras(args_):
        i, bb, tif, transform, H, W = args_
        idx = list(g_w.sindex.intersection(bb))
        sub = g_w.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
        sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
        shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
        lbl = (_rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                          fill=0, dtype="uint8")
               if shapes else np.zeros((H, W), dtype=np.uint8))
        return i, (tif, transform, H, W, lbl)

    with ThreadPoolExecutor(max_workers=len(dl_results)) as ex:
        for i, payload in ex.map(_ras, dl_results):
            region_labels[i] = payload
    covs = [(region_labels[i][-1] > 0).mean() * 100 for i in range(len(dl_results))]
    print(f"  coverage mean={np.mean(covs):.1f}% min={np.min(covs):.1f}% max={np.max(covs):.1f}%")

    # 4) Extract DINOv2 features (single load, sequential per region) ──
    print(f"\n[4] Extracting DINOv2 features @ {args.image_size}px")
    dino = DinoExtractorHiRes(image_size=args.image_size)
    import rasterio
    train_X, train_y = [], []
    test_data = None
    t0 = time.time()
    for i in sorted(region_labels):
        tif, transform, H, W, lbl = region_labels[i]
        with rasterio.open(tif) as src:
            bands = src.read(out_dtype="uint8")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        X, y, (Ph, Pw), feat_grid, y_grid = extract_patch_samples(rgb, lbl, dino)
        if i < len(train_bboxes):
            train_X.append(X); train_y.append(y)
        else:
            test_data = (rgb, lbl, feat_grid, y_grid, Ph, Pw, H, W)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"  TOTAL train: {len(train_X)} patches in {time.time()-t0:.1f}s "
          f"({len(train_X)/(time.time()-t0):.0f} patches/s)")
    print(f"  class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    # 5) Train MLP ───────────────────────────────────────────────────────
    print(f"\n[5] Training MLP")
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
                 "id_to_label": ID_TO_LABEL, "scheme": "8class"},
                args.out_dir / "head_v4.joblib")

    # 6) Evaluate on held-out test region ───────────────────────────────
    print(f"\n[6] Evaluation vs 8-class DLTB ground truth")
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
    print(f"  test acc: {acc:.3f} (n={int(valid.sum())} pixels)")
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    print(f"  per-class IoU:")
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
        print(f"    {ID_TO_LABEL.get(c, str(c)):<4} (id={c}): {ious[c]:.3f}")
    print(f"  macro IoU: {np.mean(list(ious.values())):.3f}")

    # Save the prediction.
    test_tif = region_labels[len(train_bboxes)][0]
    with rasterio.open(test_tif) as src:
        prof = src.profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0)
    for k in ("photometric", "interleave"): prof.pop(k, None)
    with rasterio.open(args.out_dir / "test_pred.tif", "w", **prof) as dst:
        dst.write(pred_full, 1)
    print(f"\n  saved → {args.out_dir / 'test_pred.tif'}")


if __name__ == "__main__":
    main()
