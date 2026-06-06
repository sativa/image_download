"""End-to-end benchmark against 三调 DLTB ground truth.

Workflow:
  1. Pick a small bbox already known to contain all 5 三调 classes
     (passed in as --bbox; we found a good one earlier in Heshui).
  2. Download XYZ tiles for that bbox at z17 in parallel.
  3. Stitch the tiles and write a 4-band GeoTIFF in EPSG:3857.
  4. Rasterise the DLTB polygons (clipped to the bbox) into a
     ground-truth label raster on the same pixel grid.
  5. For every (backend, classifier) combo, run the sidecar in
     parallel, rasterise its output GPKG, compute per-pixel agreement
     vs the ground-truth raster.
  6. Print a confusion matrix and a ranked summary.

Why this matters:
  - All previous comparisons were "metric without ground truth" — we
    only knew which backends agreed, not which was correct.
  - With DLTB we can now compute accuracy / IoU / macro-F1 properly.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests


EARTH_HALF_CIRC_M = 20037508.3427892
TILE_PX = 256

# 三调 一级地类 → numeric ids the sidecar uses.
# We adopt the 5-class 三调 scheme as the canonical target — it's what
# the ground truth provides, and it maps cleanly back to our existing
# LAND_COVER constants via a small re-labelling step.
DLTB_CLASS_TO_ID = {
    "耕地": 1,
    "园地": 2,
    "林地": 3,
    "草地": 4,
    "其他": 5,
}
ID_TO_DLTB = {v: k for k, v in DLTB_CLASS_TO_ID.items()}

# How to map our sidecar's 6-class scheme onto the 三调 5-class scheme.
# This is the lens through which we evaluate "did the sidecar get it right".
SIDECAR_TO_DLTB = {
    1: 3,   # forest    → 林地
    2: 4,   # grassland → 草地
    3: 1,   # cropland  → 耕地
    4: 5,   # water     → 其他
    5: 5,   # bare_soil → 其他
    6: 5,   # built_up  → 其他
}


def lon_to_tile_x(lon: float, z: int) -> int:
    n = 2.0 ** z
    return int(math.floor((lon + 180.0) / 360.0 * n))


def lat_to_tile_y(lat: float, z: int) -> int:
    n = 2.0 ** z
    lat = max(-85.05112878, min(85.05112878, lat))
    rad = math.radians(lat)
    return int(math.floor((1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * n))


def tile_bbox_3857(x: int, y: int, z: int):
    n = 2 ** z
    cell = 2 * EARTH_HALF_CIRC_M / n
    west = -EARTH_HALF_CIRC_M + x * cell
    east = -EARTH_HALF_CIRC_M + (x + 1) * cell
    north = EARTH_HALF_CIRC_M - y * cell
    south = EARTH_HALF_CIRC_M - (y + 1) * cell
    return west, south, east, north


def download_one(session, url):
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return r.content


def download_bbox(bbox_wgs84, zoom: int, out_tif: Path):
    """Fetch all XYZ tiles for the bbox at zoom, stitch, write GeoTIFF.

    Uses Esri World Imagery; falls back per-tile to Google. Concurrent
    downloads via ThreadPoolExecutor (32 workers — fine for an M3 Ultra).
    """
    from PIL import Image
    import rasterio
    from rasterio.transform import from_origin

    w, s, e, n = bbox_wgs84
    x_min = lon_to_tile_x(w, zoom)
    x_max = lon_to_tile_x(e, zoom)
    y_min = lat_to_tile_y(n, zoom)  # north → smaller y
    y_max = lat_to_tile_y(s, zoom)
    tx_count = x_max - x_min + 1
    ty_count = y_max - y_min + 1
    print(f"  tiles: {tx_count} × {ty_count} = {tx_count*ty_count} at z{zoom}")

    canvas_w, canvas_h = tx_count * TILE_PX, ty_count * TILE_PX
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    session = requests.Session()
    session.headers["User-Agent"] = "bench/1.0 (+imagery_downloader)"

    def fetch(xy):
        x, y = xy
        urls = [
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}",
            f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}",
        ]
        for url in urls:
            try:
                return (x, y, download_one(session, url))
            except Exception:
                continue
        return (x, y, None)

    tasks = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        for x, y, data in ex.map(fetch, tasks):
            if data is None:
                continue
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            rx, ry = (x - x_min) * TILE_PX, (y - y_min) * TILE_PX
            canvas[ry:ry + TILE_PX, rx:rx + TILE_PX] = img
    print(f"  download+stitch took {time.time()-t0:.1f}s")

    # Compute the geographic transform: stitched canvas exactly covers
    # the tile-snapped bbox in EPSG:3857.
    snap_w, _, _, snap_n = tile_bbox_3857(x_min, y_min, zoom)
    snap_e, _, _, _ = tile_bbox_3857(x_max, y_min, zoom)
    _, snap_s, _, _ = tile_bbox_3857(x_min, y_max, zoom)
    px_x = (snap_e - snap_w) / canvas_w
    px_y = (snap_n - snap_s) / canvas_h
    transform = from_origin(snap_w, snap_n, px_x, px_y)
    profile = {
        "driver": "GTiff", "height": canvas_h, "width": canvas_w,
        "count": 3, "dtype": "uint8", "crs": "EPSG:3857",
        "transform": transform, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        for i in range(3):
            dst.write(canvas[..., i], i + 1)
    print(f"  wrote {out_tif} ({out_tif.stat().st_size/1024:.0f} KB)")
    return out_tif, (snap_w, snap_s, snap_e, snap_n), canvas_w, canvas_h


def rasterise_dltb(gpkg_path: Path, bbox_wgs84, raster_tif: Path, out_npy: Path):
    """Clip DLTB polygons to bbox, reproject to EPSG:3857, rasterise."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box as shapely_box

    print("  reading + clipping DLTB ...", flush=True)
    g = gpd.read_parquet(gpkg_path).to_crs("EPSG:4326")
    g = g.clip(shapely_box(*bbox_wgs84))
    print(f"  {len(g)} polygons after clip")
    g = g.to_crs("EPSG:3857")

    g["class_id"] = g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)

    with rasterio.open(raster_tif) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
    shapes = [(geom, int(cid)) for geom, cid in zip(g.geometry, g["class_id"]) if cid > 0]
    label = rasterize(shapes=shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8")
    np.save(out_npy, label)
    coverage = (label > 0).mean() * 100
    print(f"  ground-truth coverage: {coverage:.1f}%")
    return label


def run_sidecar(input_tif, backend, classifier, out_dir):
    out_path = out_dir / f"pred__{backend}__{classifier}.landform.tif"
    for sib in out_dir.glob(f"pred__{backend}__{classifier}.landform.*"):
        sib.unlink()
    cmd = [
        "/Users/zhangfeng/D/sam3/sam3_env_py312/bin/python", "-m", "sam3_classify",
        "--input", str(input_tif),
        "--output", str(out_path),
        "--weights", "/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt",
        "--device", "cpu", "--confidence", "0.05",
        "--backend", backend, "--classifier", classifier,
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd="/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar",
        capture_output=True, text=True,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONWARNINGS": "ignore"},
    )
    elapsed = time.time() - t0
    gpkg = out_path.with_suffix(".gpkg")
    return {
        "backend": backend, "classifier": classifier, "elapsed": elapsed,
        "ok": gpkg.exists() and proc.returncode == 0,
        "gpkg": str(gpkg),
        "stderr_tail": proc.stderr[-300:] if proc.returncode != 0 else "",
    }


def rasterise_prediction(gpkg_path, ref_tif):
    """Rasterise sidecar output on the same grid and map to DLTB scheme."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    g = gpd.read_file(gpkg_path, layer="landform")
    with rasterio.open(ref_tif) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
    g["dltb_id"] = g["class_id"].map(SIDECAR_TO_DLTB).fillna(0).astype(int)
    shapes = [(geom, int(cid)) for geom, cid in zip(g.geometry, g["dltb_id"])]
    return rasterize(shapes=shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8")


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Per-pixel accuracy / per-class IoU on the DLTB-labelled pixels only."""
    valid = truth > 0
    if not valid.any():
        return {"accuracy": 0.0, "iou": {}, "macro_iou": 0.0, "confusion": {}}
    p = pred[valid]
    t = truth[valid]
    acc = float((p == t).mean())
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    iou = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        iou[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(iou.values()))) if iou else 0.0
    # Confusion (truth → predicted dict)
    confusion = {}
    for tc in classes:
        row = {}
        tm = t == tc
        if not tm.any():
            confusion[ID_TO_DLTB.get(int(tc), str(tc))] = row
            continue
        for pc in classes:
            n = int(((p == pc) & tm).sum())
            row[ID_TO_DLTB.get(int(pc), str(pc))] = n
        confusion[ID_TO_DLTB.get(int(tc), str(tc))] = row
    return {"accuracy": acc, "per_class_iou": iou, "macro_iou": macro,
            "confusion": confusion}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   help="W S E N (WGS84)")
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/bench_truth"))
    p.add_argument("--combos", default="slic,color slic,siglip sam3,color dino,color",
                   help="space-separated 'backend,classifier' pairs")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1+2: download imagery ──────────────────────────────────────
    print(f"\n[1] Downloading imagery for bbox {args.bbox} at z{args.zoom}")
    input_tif = args.out_dir / f"input_z{args.zoom}.tif"
    if not input_tif.exists():
        download_bbox(args.bbox, args.zoom, input_tif)
    else:
        print(f"  using cached {input_tif}")

    # ── Step 3: ground truth ────────────────────────────────────────────
    print(f"\n[2] Rasterising DLTB ground truth")
    truth_npy = args.out_dir / "truth_label.npy"
    truth = rasterise_dltb(args.dltb, tuple(args.bbox), input_tif, truth_npy)

    # ── Step 4: run all (backend, classifier) combos in PARALLEL ────────
    combos = [c.split(",") for c in args.combos.split()]
    print(f"\n[3] Running {len(combos)} combos in parallel: {combos}")
    job_results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=len(combos)) as ex:
        futures = {
            ex.submit(run_sidecar, input_tif, b, c, args.out_dir): (b, c)
            for b, c in combos
        }
        for fut in as_completed(futures):
            b, c = futures[fut]
            r = fut.result()
            tag = "ok" if r["ok"] else "FAIL"
            print(f"  [{tag}] {b}+{c} — {r['elapsed']:.1f}s")
            job_results.append(r)
    print(f"  total wall: {time.time()-t_start:.1f}s")

    # ── Step 5: evaluate each prediction ────────────────────────────────
    print(f"\n[4] Evaluating against ground truth")
    rows = []
    for r in job_results:
        if not r["ok"]:
            rows.append({**r, "accuracy": 0.0, "macro_iou": 0.0})
            continue
        pred = rasterise_prediction(Path(r["gpkg"]), input_tif)
        metrics = evaluate(pred, truth)
        rows.append({**r, **metrics})
        print(f"  {r['backend']}+{r['classifier']}: "
              f"acc={metrics['accuracy']:.3f}  macro_iou={metrics['macro_iou']:.3f}")

    # ── Step 6: report ──────────────────────────────────────────────────
    rows.sort(key=lambda d: -d.get("accuracy", 0))
    print("\n" + "=" * 70)
    print(f"{'combo':<20} {'time(s)':<10} {'accuracy':<12} {'macro_IoU':<12}")
    print("-" * 70)
    for r in rows:
        combo = f"{r['backend']}+{r['classifier']}"
        print(f"{combo:<20} {r['elapsed']:<10.1f} "
              f"{r.get('accuracy', 0):<12.3f} {r.get('macro_iou', 0):<12.3f}")
    print("=" * 70)
    print()
    # Show the best combo's confusion matrix.
    best = rows[0]
    if best.get("ok") and best.get("confusion"):
        print(f"Confusion matrix for winner: {best['backend']}+{best['classifier']}")
        cm = best["confusion"]
        cls_names = list(cm.keys())
        print(f"  rows=truth, cols=predicted")
        print(f"  {'truth\\pred':<10} " + " ".join(f"{c:<8}" for c in cls_names))
        for tc in cls_names:
            row = cm[tc]
            print(f"  {tc:<10} " + " ".join(f"{row.get(pc, 0):<8}" for pc in cls_names))
    (args.out_dir / "report.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nReport saved to {args.out_dir / 'report.json'}")
    print(f"All intermediate artefacts in {args.out_dir}")


if __name__ == "__main__":
    main()
