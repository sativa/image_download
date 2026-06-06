"""Changzhi cross-province: fuse 1m Esri+Google -> 6ch@1m per cell (no label needed; only
for stage-1 inference). Keyed by changzhi_{r0}_{c0} to match changzhi_cells.pkl. ProcessPool."""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds


def rp(tif, dst, SZ):
    a = np.zeros((3, SZ, SZ), np.uint8)
    with rasterio.open(tif) as s:
        for b in range(min(3, s.count)):
            reproject(rasterio.band(s, b + 1), a[b], src_transform=s.transform, src_crs=s.crs,
                      dst_transform=dst, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
    return a


def worker(args):
    name, bbox, hires, out, res = args
    e = Path(hires) / f"{name}_esri.tif"; g = Path(hires) / f"{name}_google.tif"
    if not e.exists() or not g.exists():
        return (name, "missing")
    w, s, ee, n = bbox
    SZ = int(round((ee - w) * 111000 / res))
    dst = from_bounds(w, s, ee, n, SZ, SZ)
    x6 = np.concatenate([rp(e, dst, SZ), rp(g, dst, SZ)], 0)
    np.savez_compressed(Path(out) / f"{name}.npz", x6=x6, bbox=np.array(bbox))
    return (name, "ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions", default="/mnt/sda/zf/landform/data/changzhi_regions.json")
    p.add_argument("--hires", default="/mnt/sda/zf/landform/data/changzhi_hires")
    p.add_argument("--out", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--res", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    R = json.load(open(a.regions))
    tasks = [(f"{c['county']}_{c['idx']}", c["bbox"], a.hires, a.out, a.res) for c in R]
    print(f"[changzhi-fuse] {len(tasks)} cells, {a.workers} workers", flush=True)
    ok = miss = 0; man = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for nm, st in ex.map(worker, tasks):
            if st == "ok":
                ok += 1; man.append(nm)
            else:
                miss += 1
    json.dump({"cells": man}, open(Path(a.out) / "manifest.json", "w"))
    print(f"[done] changzhi-fuse ok={ok} miss={miss}", flush=True)


if __name__ == "__main__":
    main()
