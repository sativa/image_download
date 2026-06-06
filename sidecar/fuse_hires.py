"""Fuse 1m Esri RGB + 10m S2/NDVI -> 12-ch cells at a finer grid (default 5m/448px),
labels re-rasterized at the finer grid. Runs on .250.

Per cell -> fused_poc/{county}_{idx}.npz  {x12 float16 (12,SZ,SZ), label uint8 (SZ,SZ)}
Channels: [0:3]=Esri RGB/255 | [3:7]=S2 RGBNIR norm | [7]=S2 NDVI norm | [8:12]=NDVI 4yr norm.
The control experiment trains with --channels 9 (drop [0:3]=hires) vs 12 on the SAME cells.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
S2_MEAN = np.array([400, 460, 320, 1800], np.float32)
S2_STD = np.array([200, 200, 200, 700], np.float32)
NDVI_MEAN, NDVI_STD = 0.5, 0.3
EXTRA_YEARS = [2018, 2019, 2020, 2022]
DLBM_TO_CLASS = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 5,
                 "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5}


def _resize(arr, SZ):
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.from_numpy(arr.astype(np.float32))[None],
                         size=(SZ, SZ), mode="bilinear", align_corners=False)[0].numpy()


def load_county(code, dltb_dir):
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
    return g


def fuse_one(c, hires_dir, s2_dir, ndvi_dir, gdf, SZ):
    name = f'{c["county"]}_{c["idx"]}'
    esri = Path(hires_dir) / f"{name}_esri.tif"
    s2p = Path(s2_dir) / f"{name}.npz"
    ndp = Path(ndvi_dir) / f"{name}.npz"
    if not (esri.exists() and s2p.exists() and ndp.exists()):
        return None
    bbox = tuple(c["bbox"]); w, s, e, n = bbox
    dst_t = from_bounds(w, s, e, n, SZ, SZ)
    hires = np.zeros((3, SZ, SZ), np.float32)
    with rasterio.open(esri) as src:
        for b in range(3):
            reproject(source=rasterio.band(src, b + 1), destination=hires[b],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_t, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
    hires /= 255.0
    s2 = np.load(s2p)
    rgbnir = _resize(s2["rgbnir"].astype(np.float32), SZ)
    ndvi = _resize(s2["ndvi"].astype(np.float32)[None], SZ)[0]
    nd = np.load(ndp); yrs = nd["years"].tolist()
    stack = nd["ndvi_years"].astype(np.float32) / 10000.0
    yr = _resize(np.stack([stack[yrs.index(y)] for y in EXTRA_YEARS if y in yrs], 0), SZ)
    for b in range(4):
        rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
    ndvi = (ndvi - NDVI_MEAN) / NDVI_STD
    yr = (yr - NDVI_MEAN) / NDVI_STD
    x12 = np.concatenate([hires, rgbnir, ndvi[None], yr], 0).astype(np.float16)
    idx = list(gdf.sindex.intersection(bbox))
    label = np.zeros((SZ, SZ), np.uint8)
    if idx:
        sub = gdf.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bbox))
        sub = sub[~sub.geometry.is_empty]
        if len(sub):
            label = rasterize([(g, int(b)) for g, b in zip(sub.geometry, sub["bin"])],
                              out_shape=(SZ, SZ), transform=dst_t, fill=0, dtype="uint8")
    return x12, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poc-json", default=str(HOME / "data/poc800.json"))
    p.add_argument("--hires-dir", default=str(HOME / "data/hires_poc"))
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ndvi-dir", default=str(HOME / "data/v33_ndvi_multitemporal"))
    p.add_argument("--dltb-dir", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out-dir", default=str(HOME / "data/fused_poc"))
    p.add_argument("--sz", type=int, default=448)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    poc = json.loads(Path(a.poc_json).read_text())
    split, allc = {}, []
    for s in ("test", "train"):
        for c in poc.get(s, []):
            split[f'{c["county"]}_{c["idx"]}'] = s
            allc.append(c)
    byc = defaultdict(list)
    for c in allc:
        byc[c["county"]].append(c)
    manifest = {"train": [], "test": []}; done = 0; skip = 0
    for code, cells in byc.items():
        gdf = load_county(code, a.dltb_dir)
        for c in cells:
            r = fuse_one(c, a.hires_dir, a.s2_dir, a.ndvi_dir, gdf, a.sz)
            name = f'{c["county"]}_{c["idx"]}'
            if r is None:
                skip += 1; continue
            x12, label = r
            np.savez_compressed(out / f"{name}.npz", x12=x12, label=label)
            manifest[split[name]].append(name); done += 1
            if done % 100 == 0:
                print(f"  fused {done} (skip {skip})", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"[done] fused {done}, skipped {skip}; train={len(manifest['train'])} "
          f"test={len(manifest['test'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
