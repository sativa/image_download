"""Parallel loader for S2 cells + DLTB labels.

Bottleneck profile (single-threaded loader in train_v19_s2.py):
  22,646 cells × ~50ms rasterize = ~19 min.

Strategy:
  - Group cells by county
  - Each worker process loads ONE county's DLTB parquet (only once per county)
  - Workers process all cells of that county sequentially using cached sindex
  - 16-32 workers in parallel → ~1-2 min total
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS


def _rasterize_one(g_wgs84, bbox, transform_arr, H, W):
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from affine import Affine
    transform = Affine(*transform_arr.flatten()[:6])
    idx = list(g_wgs84.sindex.intersection(tuple(bbox)))
    if not idx: return np.zeros((H, W), dtype=np.uint8)
    sub = g_wgs84.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bbox))
    sub = sub[~sub.geometry.is_empty]
    if len(sub) == 0: return np.zeros((H, W), dtype=np.uint8)
    sub["bin"] = np.where((sub["cid"] == 1) | (sub["cid"] == 2), 1, 2)
    shapes = [(geom, int(c)) for geom, c in zip(sub.geometry, sub["bin"])]
    return rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8")


def _process_county(args):
    """One worker: load 1 county's DLTB once, process all its cells.

    args = (county_code, [cell_info, ...], dltb_dir_str, s2_dir_str, min_label_pixels)
    cell_info = dict with keys: county, idx, bbox
    Returns: list of (cell_dict, status_string)
    """
    county_code, cell_infos, dltb_dir, s2_dir, min_label_pixels = args
    import geopandas as gpd
    dltb_path = Path(dltb_dir) / f"{county_code}.parquet"
    if not dltb_path.exists():
        return [(None, f"missing parquet {county_code}") for _ in cell_infos]
    g = gpd.read_parquet(dltb_path)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try: g["geometry"] = g.geometry.make_valid()
    except Exception: g["geometry"] = g.geometry.buffer(0)
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
    # Force sindex build now (so it's done once)
    _ = g.sindex

    results = []
    for c in cell_infos:
        s2_path = Path(s2_dir) / f"{c['county']}_{c['idx']}.npz"
        if not s2_path.exists():
            results.append((None, "no npz")); continue
        data = np.load(s2_path)
        rgbnir = data["rgbnir"]; ndvi = data["ndvi"]
        H, W = rgbnir.shape[1], rgbnir.shape[2]
        label = _rasterize_one(g, data["bbox"], data["transform"], H, W)
        if (label > 0).sum() < min_label_pixels:
            results.append((None, "too few labels")); continue
        results.append(({"rgbnir": rgbnir, "ndvi": ndvi, "label": label,
                          "name": f"{c['county']}_{c['idx']}"}, "ok"))
    return results


def parallel_loadsplit(region_list, dltb_dir, s2_dir, min_label_pixels=100,
                       max_workers=16):
    """Load S2 cells + rasterize labels in parallel by county.

    region_list: list of dict with 'county', 'idx', 'bbox'
    Returns: (cells, n_skipped)
    """
    by_county = defaultdict(list)
    for r in region_list:
        by_county[r["county"]].append(r)
    print(f"  parallel load: {len(region_list)} cells across {len(by_county)} counties "
          f"using {max_workers} workers", flush=True)

    tasks = [(code, cells, str(dltb_dir), str(s2_dir), min_label_pixels)
             for code, cells in by_county.items()]

    all_cells = []; skipped = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        done = 0
        for results in ex.map(_process_county, tasks):
            for cell, status in results:
                if cell is None:
                    skipped += 1
                else:
                    all_cells.append(cell)
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{len(tasks)} counties processed "
                      f"({len(all_cells)} cells, {skipped} skipped)", flush=True)
    return all_cells, skipped


if __name__ == "__main__":
    # Quick benchmark
    import json, time, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    regions = json.loads(args.regions_json.read_text())
    t0 = time.time()
    cells, skipped = parallel_loadsplit(
        regions["train"], HOME / "data/v11_dltb", HOME / "data/v19_s2_raw",
        max_workers=args.workers,
    )
    print(f"\n[done] loaded {len(cells)} cells, skipped {skipped} in {time.time()-t0:.1f}s",
          flush=True)
