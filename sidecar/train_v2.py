"""Iteration v2: more training data + multiple classifier heads in parallel.

Improvements over train_supervised.py:
  - 12 training regions (was 4) → 3× the labelled patches.
  - 4 classifier heads trained + evaluated in parallel:
       * LR-light    : LogisticRegression(C=1.0)        (baseline)
       * LR-strong   : LogisticRegression(C=0.01)       (stronger reg)
       * LR-pca64    : PCA(64) → LR                     (low-dim, less overfit)
       * RF          : RandomForestClassifier(500)      (high-dim, non-linear)
       * MLP         : MLPClassifier(256,128)           (non-linear, regularised)
  - Same held-out test region as v1, no leakage.

Why these specific heads:
  LR-strong & LR-pca64 attack the 1024-feature / 992-sample overfit
  from v1 (train acc 100, test acc 27). RF doesn't care about
  dimensionality; MLP gives non-linear capacity to map similar-looking
  classes apart. Random forest also gives feature-importance for free.
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


# Reuse the helpers from train_supervised.
sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_supervised import (
    DLTB_CLASS_TO_ID, ID_TO_DLTB, download_region, rasterise_dltb,
    DinoExtractor, extract_patch_samples, _tile_xy, _tile_bbox_3857,
)


# 12 small bboxes spread across Heshui county. Picked from the earlier
# cell-grid scan; all have ≥4 classes within. Last entry is the test
# bbox — kept identical to v1 for direct comparison.
TRAIN_BBOXES = [
    (107.9831, 35.7923, 108.0031, 35.8123),
    (107.9031, 35.8523, 107.9231, 35.8723),
    (108.0431, 35.6923, 108.0631, 35.7123),
    (108.1031, 35.8523, 108.1231, 35.8723),
    # 8 additional cells from earlier scan (all known to span ≥4 classes).
    (107.92, 35.79, 107.94, 35.81),
    (108.00, 35.85, 108.02, 35.87),
    (108.04, 35.78, 108.06, 35.80),
    (108.06, 35.85, 108.08, 35.87),
    (107.99, 35.74, 108.01, 35.76),
    (108.12, 35.79, 108.14, 35.81),
    (108.04, 35.92, 108.06, 35.94),
    (107.88, 35.92, 107.90, 35.94),
]
TEST_BBOX = (107.8631, 35.7523, 107.8831, 35.7723)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v2"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "trainv2/1.0"

    # 1) Download all regions in parallel (12 train + 1 test) ────────
    all_bboxes = TRAIN_BBOXES + [TEST_BBOX]
    paths = [args.out_dir / f"region_{i}.tif" for i in range(len(all_bboxes))]
    print(f"[1] Downloading {len(all_bboxes)} regions in parallel")

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
    print(f"  done in {time.time()-t0:.1f}s")

    # 2) Rasterise DLTB ───────────────────────────────────────────────
    # Load the full geoparquet ONCE and fix invalid geometries with
    # buffer(0). With 13 regions each reading the 107 MB file we'd be
    # redoing 1.4 GB of I/O; the topology errors come from the original
    # geometries (DLTB has some self-intersecting polygons).
    print(f"\n[2] Rasterising DLTB for each region (load once, fix, clip in parallel)")
    import geopandas as gpd
    from shapely.geometry import box as shp_box
    from rasterio.features import rasterize as _rasterize
    t0 = time.time()
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    # `make_valid` is the modern shapely fix; fall back to buffer(0) on
    # older versions.
    try:
        full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError:
        full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)
    print(f"  loaded + fixed {len(full_g)} polygons in {time.time()-t0:.1f}s")

    region_labels = {}

    def _ras(args_):
        i, bb, tif, transform, H, W = args_
        # Spatial pre-filter via sindex — much faster than full .clip()
        # on the giant GDF.
        idx = list(full_g.sindex.intersection(bb))
        sub = full_g.iloc[idx]
        # Intersect each candidate polygon with the bbox.
        bbox_geom = shp_box(*bb)
        sub = sub.copy()
        sub["geometry"] = sub.geometry.intersection(bbox_geom)
        sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
        shapes = [(geom, int(cid)) for geom, cid in zip(sub.geometry, sub["cid"]) if cid > 0]
        lbl = _rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                         fill=0, dtype="uint8") if shapes else np.zeros((H, W), dtype=np.uint8)
        return i, (tif, transform, H, W, lbl)

    with ThreadPoolExecutor(max_workers=len(dl_results)) as ex:
        for i, payload in ex.map(_ras, dl_results):
            region_labels[i] = payload
            cov = (payload[-1] > 0).mean() * 100
            print(f"  region {i}: coverage={cov:.1f}%")

    # 3) Extract DINOv2 patch features (single model, sequential) ─────
    print(f"\n[3] Extracting features (DINOv2 single load, sequential per region)")
    dino = DinoExtractor()
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
        if i < len(TRAIN_BBOXES):
            train_X.append(X); train_y.append(y)
        else:
            test_data = (rgb, lbl, feat_grid, y_grid, Ph, Pw, H, W)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"  TOTAL train: {len(train_X)} patches across {len(TRAIN_BBOXES)} regions in "
          f"{time.time()-t0:.1f}s")
    print(f"  train class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    # 4) Train multiple heads IN PARALLEL ──────────────────────────────
    print(f"\n[4] Training 5 classifier heads in parallel")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler().fit(train_X)
    Xs = scaler.transform(train_X)

    rgb_test, lbl_test, feat_grid_test, y_grid_test, Ph, Pw, H, W = test_data
    D = feat_grid_test.shape[-1]
    Xall_test = feat_grid_test.reshape(-1, D)
    Xall_test_s = scaler.transform(Xall_test)

    def _train_head(name, build_fn, fit_extra=None):
        t = time.time()
        clf = build_fn()
        if fit_extra:
            X_in = fit_extra["fit_X"]
            Xte_in = fit_extra["test_X"]
        else:
            X_in = Xs
            Xte_in = Xall_test_s
        clf.fit(X_in, train_y)
        train_acc = clf.score(X_in, train_y)
        pred_grid = clf.predict(Xte_in).reshape(Ph, Pw)
        return name, {
            "elapsed": time.time() - t,
            "train_acc": train_acc,
            "pred_grid": pred_grid,
        }

    # Precompute PCA-reduced features (cannot be in lambda — sklearn fit is slow).
    pca = PCA(n_components=64).fit(Xs)
    Xs_pca = pca.transform(Xs)
    Xall_test_pca = pca.transform(Xall_test_s)

    head_specs = [
        ("LR-light", lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0), None),
        ("LR-strong", lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=0.01), None),
        ("LR-pca64", lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
         {"fit_X": Xs_pca, "test_X": Xall_test_pca}),
        ("RF", lambda: RandomForestClassifier(n_estimators=500, max_depth=15, class_weight="balanced",
                                              n_jobs=-1, random_state=0), None),
        ("MLP", lambda: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                                       early_stopping=True, random_state=0), None),
    ]
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=len(head_specs)) as ex:
        for name, payload in ex.map(lambda spec: _train_head(*spec), head_specs):
            results[name] = payload
    print(f"  all heads trained in {time.time()-t0:.1f}s wall")

    # 5) Evaluate each head ────────────────────────────────────────────
    print(f"\n[5] Pixel-level evaluation vs DLTB ground truth")
    valid_mask = lbl_test > 0
    print(f"  evaluating on {int(valid_mask.sum())} labelled pixels ({valid_mask.mean()*100:.1f}% of test)")

    summary = []
    for name, r in results.items():
        pred_full = np.zeros((H, W), dtype=np.uint8)
        for i in range(Ph):
            y0 = int(i * H / Ph); y1 = int((i + 1) * H / Ph)
            for j in range(Pw):
                x0 = int(j * W / Pw); x1 = int((j + 1) * W / Pw)
                pred_full[y0:y1, x0:x1] = r["pred_grid"][i, j]
        p = pred_full[valid_mask]
        t = lbl_test[valid_mask]
        acc = float((p == t).mean())
        classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
        classes = [c for c in classes if c != 0]
        ious = {}
        for c in classes:
            inter = int(((p == c) & (t == c)).sum())
            union = int(((p == c) | (t == c)).sum())
            ious[int(c)] = inter / union if union else 0.0
        macro = float(np.mean(list(ious.values()))) if ious else 0.0
        summary.append((name, r["elapsed"], r["train_acc"], acc, macro, ious))

    summary.sort(key=lambda x: -x[3])
    print()
    print(f"  {'head':<12} {'train_t(s)':<12} {'train_acc':<12} {'test_acc':<12} {'macro_IoU':<12}")
    print("  " + "-" * 65)
    for name, t, ta, acc, miou, _ in summary:
        print(f"  {name:<12} {t:<12.1f} {ta:<12.3f} {acc:<12.3f} {miou:<12.3f}")
    print()
    best_name, _, _, best_acc, best_miou, best_ious = summary[0]
    print(f"WINNER: {best_name}  acc={best_acc:.3f}  macro_IoU={best_miou:.3f}")
    print("  per-class IoU:")
    for cid, iou in best_ious.items():
        print(f"    {ID_TO_DLTB.get(cid, str(cid)):<6}: {iou:.3f}")


if __name__ == "__main__":
    main()
