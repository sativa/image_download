"""Build refined 8-class land-COVER labels using DLTB SECOND-level codes (DLBM 4-digit), to fix the
heterogeneous '其他土地(12)' bucket.

Classes (0 = nodata):
  1 耕地(01)  2 园地(02)  3 林地(03)  4 草地(04)  5 水体(11)
  6 建筑(05-10 商服/工矿/住宅/公管/特殊/交通)
  7 荒漠 (12 except 1202: 空闲地/盐碱地/沙地/裸土地/裸岩石砾地 — true bare)
  8 设施大棚 (1202 设施农用地 — reflective greenhouses/sheds, visually distinct; split out of 荒漠/建筑)

One {name}.npy (uint8) per cell, rasterized from the DLTB polygons. ProcessPool over cells.
"""
import argparse, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

DLTB = "/home/ps/landform/data/v11_dltb"
DATA = "/mnt/sda/zf/landform/data/c_1m"
OUT = "/mnt/sda/zf/landform/data/c_1m_label8"
_cache = {}


def cls_from_dlbm(dlbm_series):
    d4 = dlbm_series.astype(str).str.zfill(4)
    p2 = d4.str[:2].values
    cid = np.zeros(len(d4), np.int64)
    cid[p2 == "01"] = 1
    cid[p2 == "02"] = 2
    cid[p2 == "03"] = 3
    cid[p2 == "04"] = 4
    cid[p2 == "11"] = 5
    cid[np.isin(p2, ["05", "06", "07", "08", "09", "10"])] = 6
    cid[p2 == "12"] = 7                       # 其他土地 -> 荒漠 (true bare)
    cid[d4.values == "1202"] = 8              # 设施农用地 -> 设施大棚 (override)
    return cid


def load_county(county):
    if county in _cache:
        return _cache[county]
    g = gpd.read_parquet(Path(DLTB) / f"{county}.parquet")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except Exception:
        g["geometry"] = g.geometry.buffer(0)
    g["cid"] = cls_from_dlbm(g["DLBM"])
    g = g[g["cid"] >= 1].reset_index(drop=True)
    _cache[county] = g
    return g


def one(name):
    outp = Path(OUT) / f"{name}.npy"
    if outp.exists():
        return "skip"
    try:
        z = np.load(Path(DATA) / f"{name}.npz"); bbox = z["bbox"]; H, W = z["x6"].shape[1:]
    except Exception as e:
        return f"BAD npz {name}: {e}"
    county = name.split("_")[0]
    try:
        g = load_county(county)
    except Exception as e:
        return f"BAD county {county}: {e}"
    tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
    idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
    lab = np.zeros((H, W), np.uint8)
    if idx:
        cb = shp_box(*bbox); sub = g.iloc[idx]
        shapes = [(geom, int(c)) for geom, c in zip(sub.geometry, sub["cid"]) if geom.intersects(cb)]
        if shapes:
            lab = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
    np.save(outp, lab)
    return "ok"


def main():
    global DLTB, DATA, OUT
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--dltb", default=DLTB)
    p.add_argument("--out", default=OUT)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    DLTB = a.dltb; DATA = a.data_dir; OUT = a.out
    Path(OUT).mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(DATA) / "manifest.json").read_text())
    names = sorted(man["train"] + man["test"], key=lambda n: n.split("_")[0])  # county-sorted
    print(f"[label8] {len(names)} cells -> {OUT}", flush=True)
    ok = bad = 0; baddies = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, names, chunksize=4)):
            if r in ("ok", "skip"):
                ok += 1
            else:
                bad += 1; baddies.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(names)} ok={ok} bad={bad}", flush=True)
    print(f"[label8] done ok={ok} bad={bad}", flush=True)
    if baddies:
        print("  bad sample:", baddies[:5], flush=True)


if __name__ == "__main__":
    main()
