"""Build a z18 (~0.4 m/px) training set in the c_1m npz format, for fine-tuning the z17 model to
finer resolution. Per cell: esri+google z18 GeoTIFFs -> 6ch (google resampled to esri grid);
DLTB polygons -> 3-class label (0 nodata / 1 cropland(耕地+园地) / 2 non-cropland) at the z18 grid.
Saves {x6, label, bbox} per cell + a manifest.json{train,test}. ProcessPool over cells (CPU only).
"""
import argparse, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

DLTB = "/home/ps/landform/data/v11_dltb"
Z18 = "/mnt/sda/zf/landform/data/z18_test"
OUT = "/mnt/sda/zf/landform/data/c_1m_z18"
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
    cid = g["DLBM"].astype(str).str[:2]
    g["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0").astype(int)
    g = g[(g["cid"] >= 1) & (g["cid"] <= 12)].reset_index(drop=True)
    _cache[county] = g
    return g


def one(rec):
    name = f'{rec["county"]}_{rec["idx"]}'; bbox = rec["bbox"]
    ep = Path(Z18) / f"{name}_esri.tif"; gp = Path(Z18) / f"{name}_google.tif"
    outp = Path(OUT) / f"{name}.npz"
    if outp.exists():
        return f"skip {name}"
    if not ep.exists() or not gp.exists():
        return f"MISS tif {name}"
    e = rasterio.open(ep).read()[:3]; H, W = e.shape[1:]
    g = rasterio.open(gp).read()[:3]
    if g.shape[1:] != (H, W):
        g = F.interpolate(torch.from_numpy(g.astype(np.float32))[None], size=(H, W),
                          mode="bilinear", align_corners=False)[0].clamp(0, 255).numpy()
    x6 = np.concatenate([e, g], 0).astype(np.uint8)
    gc = load_county(rec["county"])
    tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
    idx = list(gc.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
    lab = np.zeros((H, W), np.uint8)
    if idx:
        cb = shp_box(*bbox); sub = gc.iloc[idx]
        shapes = [(geom, 1 if int(c) in (1, 2) else 2) for geom, c in zip(sub.geometry, sub["cid"]) if geom.intersects(cb)]
        if shapes:
            lab = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
    np.savez_compressed(outp, x6=x6, label=lab, bbox=np.array(bbox, np.float64))
    return f"ok {name} {H}x{W} crop%={100*(lab==1).mean():.0f}"


def main():
    global Z18, OUT
    p = argparse.ArgumentParser()
    p.add_argument("--regions", default="/mnt/sda/zf/landform/data/z18_ft_regions.json")
    p.add_argument("--tif-dir", default=Z18)
    p.add_argument("--out", default=OUT)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    Z18 = a.tif_dir; OUT = a.out
    Path(OUT).mkdir(parents=True, exist_ok=True)
    R = json.loads(Path(a.regions).read_text())
    tr, te = R["train"], R["test"]
    allc = sorted(tr + te, key=lambda r: r["county"])  # county-sorted -> worker cache locality
    print(f"[z18-build] {len(allc)} cells (train {len(tr)} / test {len(te)}) -> {OUT}", flush=True)
    ok = 0; miss = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, allc, chunksize=4)):
            if r.startswith("ok") or r.startswith("skip"): ok += 1
            else: miss += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(allc)} ok={ok} miss={miss}", flush=True)
    names = lambda lst: [f'{r["county"]}_{r["idx"]}' for r in lst if (Path(OUT) / f'{r["county"]}_{r["idx"]}.npz').exists()]
    json.dump({"train": names(tr), "test": names(te)}, open(Path(OUT) / "manifest.json", "w"))
    print(f"[z18-build] done ok={ok} miss={miss}; manifest train={len(names(tr))} test={len(names(te))}", flush=True)


if __name__ == "__main__":
    main()
