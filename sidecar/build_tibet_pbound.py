"""Build FSDA parcel-BOUNDARY labels for the Tibet cells (all-FSDA-parcel edges, like make_pbound but
from the FSDA shapefile) -> c_1m_tibet_pbound. Adding these to boundary-head training teaches the model
Tibet's fine fragmented field delineation (the fix for DA/our-head failing zero-shot on Tibet)."""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box as shp_box

TIF = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/z17_tibet")
SHP = "/Users/zhangfeng/Downloads/xizang_fsda/Xizang cropland datasets/ALL_Xizang_Albers.shp"
REG = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/tibet_regions.json"
OUT = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/c_1m_tibet_pbound"); OUT.mkdir(exist_ok=True)


def edges(idm):
    b = np.zeros(idm.shape, bool)
    d = idm[:-1, :] != idm[1:, :]; b[:-1, :] |= d; b[1:, :] |= d
    d = idm[:, :-1] != idm[:, 1:]; b[:, :-1] |= d; b[:, 1:] |= d
    out = b.copy()                                             # dilate 1px -> ~2-3px band (match make_pbound)
    out[:-1, :] |= b[1:, :]; out[1:, :] |= b[:-1, :]; out[:, :-1] |= b[:, 1:]; out[:, 1:] |= b[:, :-1]
    return out.astype(np.uint8)


print("loading FSDA -> 3857 ...", flush=True)
fsda = gpd.read_file(SHP).to_crs("EPSG:3857"); sidx = fsda.sindex
regions = json.load(open(REG)); n = 0
for r in regions:
    nm = f"XZ_{r['idx']}"; ep = TIF / f"{nm}_esri.tif"
    if not ep.exists():
        continue
    with rasterio.open(ep) as s:
        H, W = s.height, s.width; tr = s.transform; bn = s.bounds
    idx = list(sidx.intersection((bn.left, bn.bottom, bn.right, bn.top)))
    idm = np.zeros((H, W), np.int32)
    if idx:
        sub = fsda.iloc[idx].reset_index(drop=True); cb = shp_box(bn.left, bn.bottom, bn.right, bn.top)
        shp = [(g, j + 1) for j, g in enumerate(sub.geometry) if g.intersects(cb)]
        if shp:
            idm = rasterize(shp, out_shape=(H, W), transform=tr, fill=0, dtype="int32")
    np.save(OUT / f"{nm}.npy", edges(idm)); n += 1
    if n % 30 == 0:
        print(f"  {n} cells", flush=True)
print(f"done: {n} Tibet boundary labels -> {OUT}", flush=True)
