"""Route (c) — Stage-1 data prep (PARALLEL by county): per cell, reproject Esri + Google
1m RGB onto a common 1m WGS84 grid (multi-source = 6 channels) + rasterize DLTB cropland at 1m.

Multi-source: Esri & Google are two independent captures -> a real boundary appears in BOTH
(robust); artifacts/date-mismatch appear in one (filtered) -> cleaner 1m boundaries for stage-1.

Output per cell: c_1m/{county}_{idx}.npz  {x6 uint8 (6,SZ,SZ), label uint8 (SZ,SZ), bbox}
SZ = cell span / res_m (~2200 px @ 1 m for a 0.02deg cell).
"""
import argparse, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
DLBM_TO_CLASS = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 5,
                 "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5}


def _county_worker(args):
    code, cells, hires_dir, dltb_dir, out_dir, res_m, require_both = args
    try:
        g = gpd.read_parquet(Path(dltb_dir) / f"{code}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        try:
            g["geometry"] = g.geometry.make_valid()
        except Exception:
            g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        g = g[g["cid"] > 0].copy()
        g["bin"] = np.where(g["cid"].isin([1, 2]), 1, 2)
        _ = g.sindex
    except Exception as ex:
        return [(c["county"], c["idx"], f"county-err:{str(ex)[:40]}") for c in cells]

    def rp(tif, dst, SZ):
        a = np.zeros((3, SZ, SZ), np.uint8)
        with rasterio.open(tif) as src:
            for b in range(min(3, src.count)):
                reproject(rasterio.band(src, b + 1), a[b], src_transform=src.transform,
                          src_crs=src.crs, dst_transform=dst, dst_crs="EPSG:4326",
                          resampling=Resampling.bilinear)
        return a

    out = []
    for c in cells:
        name = f'{c["county"]}_{c["idx"]}'
        esri = Path(hires_dir) / f"{name}_esri.tif"
        goog = Path(hires_dir) / f"{name}_google.tif"
        if not esri.exists() or (require_both and not goog.exists()):
            out.append((c["county"], c["idx"], "missing")); continue
        w, s, e, n = c["bbox"]
        SZ = int(round((e - w) * 111000 / res_m))
        dst = from_bounds(w, s, e, n, SZ, SZ)
        x6 = np.concatenate([rp(esri, dst, SZ),
                             rp(goog, dst, SZ) if goog.exists() else np.zeros((3, SZ, SZ), np.uint8)], 0)
        idx = list(g.sindex.intersection((w, s, e, n)))
        label = np.zeros((SZ, SZ), np.uint8)
        if idx:
            sub = g.iloc[idx].copy()
            sub["geometry"] = sub.geometry.intersection(shp_box(w, s, e, n))
            sub = sub[~sub.geometry.is_empty]
            if len(sub):
                label = rasterize([(gg, int(b)) for gg, b in zip(sub.geometry, sub["bin"])],
                                  out_shape=(SZ, SZ), transform=dst, fill=0, dtype="uint8")
        np.savez_compressed(Path(out_dir) / f"{name}.npz", x6=x6, label=label, bbox=np.array(c["bbox"]))
        out.append((c["county"], c["idx"], "ok"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", default=str(HOME / "data/v40_5k.json"))
    p.add_argument("--hires-dir", default=str(HOME / "data/hires_full"))
    p.add_argument("--dltb-dir", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out-dir", default=str(HOME / "data/c_1m"))
    p.add_argument("--res-m", type=float, default=1.0)
    p.add_argument("--require-both", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    R = json.loads(Path(a.regions_json).read_text())
    split = {}
    byc = defaultdict(list)
    for s in ("test", "train"):
        for c in R.get(s, []):
            split[f'{c["county"]}_{c["idx"]}'] = s
            byc[c["county"]].append(c)
    tasks = [(code, cells, a.hires_dir, a.dltb_dir, str(out), a.res_m, a.require_both)
             for code, cells in byc.items()]
    print(f"[fuse1m] {sum(len(v) for v in byc.values())} cells / {len(tasks)} counties, {a.workers} workers", flush=True)
    man = {"train": [], "test": []}; done = 0; skip = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        nc = 0
        for res in ex.map(_county_worker, tasks):
            nc += 1
            for county, idx, st in res:
                name = f"{county}_{idx}"
                if st == "ok":
                    man[split[name]].append(name); done += 1
                else:
                    skip += 1
            if nc % 10 == 0:
                print(f"  {nc}/{len(tasks)} counties ({done} ok, {skip} skip)", flush=True)
    (out / "manifest.json").write_text(json.dumps(man))
    print(f"[done] 1m-fuse {done}, skipped {skip}; train={len(man['train'])} test={len(man['test'])}", flush=True)


if __name__ == "__main__":
    main()
