"""Parallel by-county loader for multi-temporal training (v33/v34c/v35).

Adds NDVI year stack to the cells loaded by fast_load_s2-style worker.

Each cell stored as dict: {rgbnir, ndvi_s2, ndvi_years, label, name}
ndvi_years already upsampled to S2 grid (Hs × Ws).

Speedup: 22K cells single-thread loadsplit ~28 min → parallel ~1-2 min.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS


def _upsample_to(arr2d, H, W):
    """Bilinear upsample via torch (small overhead but stable)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(arr2d.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def _rasterize_label(g_wgs84, bbox, transform_arr, H, W):
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
    """One worker: 1 county DLTB + all its cells with multi-temporal stack."""
    (county_code, cell_infos, dltb_dir, s2_dir, ndvi_yr_dir,
     extra_years, min_label_pixels) = args
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
    _ = g.sindex  # build once

    results = []
    for c in cell_infos:
        s2_path = Path(s2_dir) / f"{c['county']}_{c['idx']}.npz"
        nd_path = Path(ndvi_yr_dir) / f"{c['county']}_{c['idx']}.npz"
        if not s2_path.exists() or not nd_path.exists():
            results.append((None, "no npz")); continue
        s2 = np.load(s2_path)
        nd = np.load(nd_path)
        rgbnir = s2["rgbnir"]
        ndvi_s2 = s2["ndvi"]
        Hs, Ws = rgbnir.shape[1], rgbnir.shape[2]
        years = nd["years"].tolist()
        stack = nd["ndvi_years"].astype(np.float32) / 10000.0
        stack_sel = np.stack([stack[years.index(y)] for y in extra_years if y in years], 0)
        ndvi_years_up = np.stack([_upsample_to(stack_sel[i], Hs, Ws)
                                   for i in range(len(stack_sel))], 0).astype(np.float32)
        label = _rasterize_label(g, s2["bbox"], s2["transform"], Hs, Ws)
        if (label > 0).sum() < min_label_pixels:
            results.append((None, "few labels")); continue
        results.append(({
            "rgbnir": rgbnir, "ndvi_s2": ndvi_s2,
            "ndvi_years": ndvi_years_up, "label": label,
            "name": f"{c['county']}_{c['idx']}",
        }, "ok"))
    return results


def parallel_loadsplit_multitemp(region_list, dltb_dir, s2_dir, ndvi_yr_dir,
                                   extra_years, min_label_pixels=100,
                                   max_workers=16):
    """Parallel by-county loader with NDVI year stack.

    Returns (cells, skipped_count).
    """
    by_county = defaultdict(list)
    for r in region_list:
        by_county[r["county"]].append(r)
    print(f"  parallel load (multitemp): {len(region_list)} cells across "
          f"{len(by_county)} counties using {max_workers} workers", flush=True)

    tasks = [(code, cells, str(dltb_dir), str(s2_dir), str(ndvi_yr_dir),
              extra_years, min_label_pixels)
             for code, cells in by_county.items()]
    all_cells = []; skipped = 0
    import time
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        done = 0
        for results in ex.map(_process_county, tasks):
            for cell, status in results:
                if cell is None: skipped += 1
                else: all_cells.append(cell)
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{len(tasks)} counties ({len(all_cells)} cells, "
                      f"{skipped} skipped, {time.time()-t0:.0f}s)", flush=True)
    print(f"  load total {time.time()-t0:.0f}s, {len(all_cells)} cells", flush=True)
    return all_cells, skipped


if __name__ == "__main__":
    import json, argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--years", type=str, default="2018,2019,2020,2022",
                   help="comma-separated NDVI years")
    p.add_argument("--ndvi-yr-dir", type=Path, default=HOME / "data/v33_ndvi_multitemporal")
    args = p.parse_args()
    extra_years = [int(y) for y in args.years.split(",")]
    regions = json.loads(args.regions_json.read_text())
    print(f"benchmark on {len(regions['train'])} cells, years={extra_years}", flush=True)
    cells, skipped = parallel_loadsplit_multitemp(
        regions["train"], HOME / "data/v11_dltb", HOME / "data/v19_s2_raw",
        args.ndvi_yr_dir, extra_years, max_workers=args.workers,
    )
    print(f"\n[done] {len(cells)} cells, {skipped} skipped", flush=True)
