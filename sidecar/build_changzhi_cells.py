"""Build Changzhi (Shanxi) CROSS-PROVINCE test cells in the Gansu v19/v33 9-ch format.

Verified-compatible sources (all on .174 / PSStore):
  RGBNIR  : tempdata/changzhi_10m/opt_sentinal_annual.tif  (4-band uint16, same S2 scale as Gansu)
  S2 NDVI : tempdata/changzhi_10m/opt_sentinel_ndvi.tif    (uint8 -> /255 == Gansu float NDVI)
  yr NDVI : China_NDVI/{2018,2019,2020,2022}/NDVImax{y}.tif (int16 *10000, same archive as Gansu)
  labels  : gs_landuse/长治市_DLTB_WGS84.parquet           (DLBM -> binary 耕地+园地 vs other)

Output: changzhi_cells.pkl = list of {rgbnir, ndvi_s2, ndvi_years, label, name}
identical to fast_load_multitemp cell dicts -> scp to .250, eval with eval_xcounty.py --cells-pkl.
"""
import itertools
import pickle

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, transform as win_transform
from rasterio.enums import Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box as shp_box

ZF = "/mnt/sdb/shared/zf"
RGB_TIF = f"{ZF}/tempdata/changzhi_10m/opt_sentinal_annual.tif"
NDVI_TIF = f"{ZF}/tempdata/changzhi_10m/opt_sentinel_ndvi.tif"
DLTB_PQ = f"{ZF}/gs_landuse/长治市_DLTB_WGS84.parquet"
YEARS = [2018, 2019, 2020, 2022]
YR_TIF = {y: f"{ZF}/China_NDVI/{y}/NDVImax{y}.tif" for y in YEARS}
NDVI_NODATA = 32767
DLBM_TO_CLASS = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 5,
                 "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5}
CELL, STRIDE, MAX_CELLS = 240, 240, 160
MIN_VALID, MIN_FRAC = 8000, 0.08
OUT = f"{ZF}/tempdata/changzhi_cells.pkl"


def main():
    rgb_ds = rasterio.open(RGB_TIF)
    ndvi_ds = rasterio.open(NDVI_TIF)
    W, H, T = rgb_ds.width, rgb_ds.height, rgb_ds.transform
    print(f"changzhi raster {W}x{H}", flush=True)

    print("loading Changzhi DLTB (840k polys)...", flush=True)
    g = gpd.read_parquet(DLTB_PQ, columns=["DLBM", "geometry"])
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    try:
        g["geometry"] = g.geometry.make_valid()
    except Exception:
        g["geometry"] = g.geometry.buffer(0)
    g = g.assign(cid=g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int))
    g = g[g["cid"] > 0].copy()
    g["binc"] = np.where(g["cid"].isin([1, 2]), 1, 2)
    sidx = g.sindex
    print(f"  {len(g)} cropland-relevant polys", flush=True)

    yr_ds = {y: rasterio.open(p) for y, p in YR_TIF.items()}

    grid = list(itertools.product(range(0, H - CELL, STRIDE), range(0, W - CELL, STRIDE)))
    step = max(1, len(grid) // (MAX_CELLS * 8))
    cells = []
    for (r0, c0) in grid[::step]:
        if len(cells) >= MAX_CELLS:
            break
        win = Window(c0, r0, CELL, CELL)
        rgb = rgb_ds.read(window=win)
        if rgb.shape != (4, CELL, CELL):
            continue
        if int((rgb[3] > 0).sum()) < CELL * CELL * 0.6:      # mostly nodata (NIR band)
            continue
        nd = ndvi_ds.read(1, window=win).astype(np.float32) / 255.0
        wt = win_transform(win, T)
        minx, maxy = wt * (0, 0)
        maxx, miny = wt * (CELL, CELL)
        bbox = (minx, miny, maxx, maxy)
        cand = list(sidx.intersection(bbox))
        if not cand:
            continue
        sub = g.iloc[cand]
        sub = sub[sub.geometry.intersects(shp_box(*bbox))]
        if len(sub) == 0:
            continue
        label = rasterize([(geom, int(b)) for geom, b in zip(sub.geometry, sub["binc"])],
                          out_shape=(CELL, CELL), transform=wt, fill=0, dtype="uint8")
        nv = int((label > 0).sum())
        if nv < MIN_VALID:
            continue
        frac = float((label == 1).sum()) / nv
        if frac < MIN_FRAC or frac > 1 - MIN_FRAC:           # need both classes for meaningful F1
            continue
        yrs = []
        for y in YEARS:
            ds = yr_ds[y]
            wy = from_bounds(minx, miny, maxx, maxy, ds.transform)
            arr = ds.read(1, window=wy, out_shape=(CELL, CELL),
                          resampling=Resampling.bilinear).astype(np.float32)
            arr[arr == NDVI_NODATA] = np.nan
            fill = np.nanmedian(arr) if np.isfinite(arr).any() else 5000.0
            arr = np.where(np.isfinite(arr), arr, fill) / 10000.0
            yrs.append(arr)
        cells.append({"rgbnir": rgb.astype(np.uint16), "ndvi_s2": nd,
                      "ndvi_years": np.stack(yrs, 0).astype(np.float32),
                      "label": label, "name": f"changzhi_{r0}_{c0}"})
        if len(cells) % 20 == 0:
            print(f"  {len(cells)} cells (last crop_frac={frac:.2f})", flush=True)

    print(f"selected {len(cells)} Changzhi test cells", flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(cells, f)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
