"""Build PARCEL-BOUNDARY labels: every DLTB polygon edge (including ridges between adjacent
same-class fields), so a boundary head can DELINEATE individual parcels (not just cropland outline).

Per cell: rasterize each DLTB polygon to a unique id, mark pixels where the id changes (4-neighbour)
as boundary, dilate 1 px for a usable target width. One {name}.npy (uint8: 1=boundary, 0=interior).
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
OUT = "/mnt/sda/zf/landform/data/c_1m_pbound"
_cache = {}


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
    _cache[county] = g
    return g


def _boundary(idm):
    b = np.zeros(idm.shape, bool)
    d = idm[:-1, :] != idm[1:, :]
    b[:-1, :] |= d; b[1:, :] |= d
    d = idm[:, :-1] != idm[:, 1:]
    b[:, :-1] |= d; b[:, 1:] |= d
    # dilate 1 px (3x3) so the target is ~2-3 px wide
    out = b.copy()
    out[:-1, :] |= b[1:, :]; out[1:, :] |= b[:-1, :]
    out[:, :-1] |= b[:, 1:]; out[:, 1:] |= b[:, :-1]
    return out.astype(np.uint8)


def one(name):
    outp = Path(OUT) / f"{name}.npy"
    if outp.exists():
        return "skip"
    try:
        z = np.load(Path(DATA) / f"{name}.npz"); bbox = z["bbox"]; H, W = z["x6"].shape[1:]
    except Exception as e:
        return f"BAD npz {name}: {e}"
    try:
        g = load_county(name.split("_")[0])
    except Exception as e:
        return f"BAD county {name}: {e}"
    tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
    idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
    idm = np.zeros((H, W), np.int32)
    if idx:
        cb = shp_box(*bbox); sub = g.iloc[idx].reset_index(drop=True)
        shapes = [(geom, j + 1) for j, geom in enumerate(sub.geometry) if geom.intersects(cb)]
        if shapes:
            idm = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="int32")
    np.save(outp, _boundary(idm))
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
    names = sorted(man["train"] + man["test"], key=lambda n: n.split("_")[0])
    print(f"[pbound] {len(names)} cells -> {OUT}", flush=True)
    ok = bad = 0; baddies = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, names, chunksize=4)):
            if r in ("ok", "skip"):
                ok += 1
            else:
                bad += 1; baddies.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(names)} ok={ok} bad={bad}", flush=True)
    print(f"[pbound] done ok={ok} bad={bad}; sample bad: {baddies[:3]}", flush=True)


if __name__ == "__main__":
    main()
