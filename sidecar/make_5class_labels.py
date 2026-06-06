"""Rasterize 5-class 1m landform labels for ALL c_1m cells from Gansu's full DLTB (三调).

c_1m's npz `label` is BINARY (0=nodata/1=cropland(耕地+园地)/2=other). For multi-class recognition we
re-rasterize the original DLTB classes at 1m: 1耕地 2园地 3林地 4草地 5其他 (0=nodata). Output one
`{name}.npy` (uint8, 2220x2220) per cell -> used by the 5-class DINOv2-1m trainer. Fully uses the
province-wide 三调 polygons (per the user: reuse all Gansu DLTB so the model fits this imagery).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS  # {'01':1,'02':2,'03':3,'04':4,'05'..'12':5}


def load_county(dltb, county, cache):
    if county in cache:
        return cache[county]
    g = gpd.read_parquet(Path(dltb) / f"{county}.parquet")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except AttributeError:
        g["geometry"] = g.geometry.buffer(0)
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
    g = g[g["cid"] > 0].reset_index(drop=True)
    cache[county] = g
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out", default="/mnt/sda/zf/landform/data/c_1m_label5")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = [n for n in (man["train"] + man["test"]) if (Path(a.data_dir) / f"{n}.npz").exists()]
    print(f"[5cls] {len(names)} cells; classes=耕地/园地/林地/草地/其他", flush=True)

    cache = {}; t0 = time.time(); done = 0; dist = np.zeros(6, np.int64)
    for n in names:
        of = out / f"{n}.npy"
        z = np.load(Path(a.data_dir) / f"{n}.npz"); bbox = z["bbox"]; H, W = z["x6"].shape[1:]
        try:
            g = load_county(a.dltb, n.split("_")[0], cache)
        except Exception as ex:
            print(f"  skip {n}: {ex}", flush=True); continue
        tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
        idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
        lab = np.zeros((H, W), np.uint8)
        if idx:
            cb = shp_box(*bbox); sub = g.iloc[idx]
            shapes = [(geom, int(c)) for geom, c in zip(sub.geometry, sub["cid"]) if geom.intersects(cb)]
            if shapes:
                lab = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
        np.save(of, lab); dist += np.bincount(lab.ravel(), minlength=6); done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    tot = dist.sum()
    print(f"[5cls] done {done} cells ({time.time()-t0:.0f}s) -> {out}", flush=True)
    names5 = ["nodata", "耕地", "园地", "林地", "草地", "其他"]
    print("  class pixel %: " + " ".join(f"{names5[i]}={dist[i]/tot*100:.1f}" for i in range(6)), flush=True)


if __name__ == "__main__":
    main()
