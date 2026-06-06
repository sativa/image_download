"""Rasterize GDLX (耕地类型) 3-class aux labels per c_1m cell, for the GDLX multi-task head.

Classes: 0 = 其他 (non-terrace/slope: flat cropland + non-cropland), 1 = 梯田 (TT), 2 = 坡地 (PD).
Source = gs_terrace_slope.parquet (TT+PD parcels with gcls{1,2}, from the full Gansu DLTB).
One {name}.npy (uint8) per cell -> c_1m_gdlx/, aligned to the cell's bbox/grid (same as its label).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--ts-parquet", default="/mnt/sda/zf/landform/data/gs_terrace_slope.parquet")
    p.add_argument("--out", default="/mnt/sda/zf/landform/data/c_1m_gdlx")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = gpd.read_parquet(a.ts_parquet)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    _ = g.sindex
    print(f"[gdlx] {len(g)} TT/PD parcels indexed ({time.time()-t0:.0f}s)", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = [n for n in (man["train"] + man["test"]) if (Path(a.data_dir) / f"{n}.npz").exists()]
    done = 0; dist = np.zeros(3, np.int64)
    for n in names:
        if (out / f"{n}.npy").exists():
            done += 1; continue
        z = np.load(Path(a.data_dir) / f"{n}.npz"); bbox = z["bbox"]; H, W = z["x6"].shape[1:]
        tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
        idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
        lab = np.zeros((H, W), np.uint8)
        if idx:
            cb = shp_box(*bbox); sub = g.iloc[idx]
            shapes = [(geom, int(c)) for geom, c in zip(sub.geometry, sub["gcls"]) if geom.intersects(cb)]
            if shapes:
                lab = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
        np.save(out / f"{n}.npy", lab); dist += np.bincount(lab.ravel(), minlength=3); done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    tot = dist.sum()
    print(f"[gdlx] done {done} -> {out} | px%: 其他={dist[0]/tot*100:.1f} 梯田={dist[1]/tot*100:.1f} 坡地={dist[2]/tot*100:.1f}", flush=True)


if __name__ == "__main__":
    main()
