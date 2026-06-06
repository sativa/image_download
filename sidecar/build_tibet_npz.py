"""Build a c_1m-format Tibet dataset for domain adaptation: x6 (esri+google) + binary cropland
label rasterized from the FSDA parcels (1 = inside an FSDA cropland parcel, 2 = outside / non-crop).
manifest: first 90 cells = adaptation (train), last 30 = held-out test."""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box as shp_box

TIF = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/z17_tibet")
SHP = "/Users/zhangfeng/Downloads/xizang_fsda/Xizang cropland datasets/ALL_Xizang_Albers.shp"
REG = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/tibet_regions.json"
OUT = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/c_1m_tibet"); OUT.mkdir(exist_ok=True)
N_TEST = 30

print("loading FSDA parcels -> 3857 ...", flush=True)
fsda = gpd.read_file(SHP).to_crs("EPSG:3857"); sidx = fsda.sindex
regions = json.load(open(REG))
names = []
for r in regions:
    nm = f"XZ_{r['idx']}"; ep = TIF / f"{nm}_esri.tif"; gp = TIF / f"{nm}_google.tif"
    if not ep.exists():
        continue
    with rasterio.open(ep) as s:
        e = s.read()[:3]; H, W = s.height, s.width; tr = s.transform; bnds = s.bounds
    g = rasterio.open(gp).read()[:3] if gp.exists() else e
    if g.shape[1:] != (H, W):
        g = F.interpolate(torch.from_numpy(g.astype(np.float32))[None], size=(H, W), mode="bilinear",
                          align_corners=False)[0].clamp(0, 255).numpy()
    x6 = np.concatenate([e, g], 0).astype(np.uint8)
    idx = list(sidx.intersection((bnds.left, bnds.bottom, bnds.right, bnds.top)))
    lab = np.full((H, W), 2, np.uint8)                      # default non-cropland
    if idx:
        cb = shp_box(bnds.left, bnds.bottom, bnds.right, bnds.top); sub = fsda.iloc[idx]
        shapes = [(geom, 1) for geom in sub.geometry if geom.intersects(cb)]
        if shapes:
            cm = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype=np.uint8)
            lab[cm == 1] = 1                                # cropland inside FSDA parcels
    np.savez_compressed(OUT / f"{nm}.npz", x6=x6, label=lab, bbox=np.array(r["bbox"], np.float64))
    names.append(nm)
    if len(names) % 20 == 0:
        print(f"  {len(names)} cells", flush=True)

man = {"train": names[:-N_TEST], "test": names[-N_TEST:]}
json.dump(man, open(OUT / "manifest.json", "w"))
print(f"done: {len(names)} cells -> c_1m_tibet | train {len(man['train'])} test {len(man['test'])}", flush=True)
