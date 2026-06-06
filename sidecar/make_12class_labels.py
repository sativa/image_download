"""Rasterize 12-class 1m labels = the full DLTB FIRST-LEVEL land classes (一级类) from Gansu DLTB.

Unlike the 5-class version (which merged 05-12 into '其他'), here cid = int(DLBM[:2]) keeps all 12
first-level classes: 1耕地 2园地 3林地 4草地 5商服 6工矿仓储 7住宅 8公管 9特殊 10交通 11水域 12其他土地
(0=nodata). One `{name}.npy` (uint8) per c_1m cell -> used by the 12-class DINOv2-1m trainer.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")


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
    # first-level class = first 2 digits of DLBM (01..12)
    cid = g["DLBM"].astype(str).str[:2]
    g["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0")
    g["cid"] = g["cid"].astype(int)
    g = g[(g["cid"] >= 1) & (g["cid"] <= 12)].reset_index(drop=True)
    cache[county] = g
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out", default="/mnt/sda/zf/landform/data/c_1m_label12")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = [n for n in (man["train"] + man["test"]) if (Path(a.data_dir) / f"{n}.npz").exists()]
    print(f"[12cls] {len(names)} cells; 12 first-level DLTB classes", flush=True)

    cache = {}; t0 = time.time(); done = 0; dist = np.zeros(13, np.int64)
    for n in names:
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
        np.save(out / f"{n}.npy", lab); dist += np.bincount(lab.ravel(), minlength=13); done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    tot = dist.sum(); nm = ["nodata", "耕地", "园地", "林地", "草地", "商服", "工矿", "住宅", "公管", "特殊", "交通", "水域", "其他"]
    print(f"[12cls] done {done} ({time.time()-t0:.0f}s) -> {out}", flush=True)
    print("  px%: " + " ".join(f"{nm[i]}={dist[i]/tot*100:.1f}" for i in range(13)), flush=True)


if __name__ == "__main__":
    main()
