"""B 省级思路曲线版长治:每块 topojson+Chaikin(块小必 fit)+ 按全局 ID dissolve(PROV 已验无缝重建,愈合块切)。
每块:vectorize(带 parcel_id)-> smooth_coverage(块内曲线)-> 质心 sjoin 回贴 parcel_id -> concat ->
dissolve by parcel_id(同一地块跨块碎片并回)-> clip + postproc。块内曲线;块边因各自独立平滑可能微错位(诚实留待 verify)。
"""
import sys, time, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio.features
from shapely.geometry import shape as _shape
from affine import Affine
import parcel_pipeline as pp
import postproc

CLASSES = [(1, "耕地", "cropland", (60, 180, 75)), (2, "园地", "orchard", (170, 255, 90)),
           (3, "林地", "forest", (0, 100, 0)), (4, "草地", "grassland", (190, 220, 100)),
           (5, "水体", "water", (0, 130, 200)), (6, "建筑", "built-up", (230, 25, 75)),
           (7, "荒漠", "bare", (170, 140, 100))]
UTM = "EPSG:32649"; TOL = 15.0; DS = 4; TILE = 4000   # 4000 /4px ≈ 19km/块,块内 topojson 县级口径必 fit
VRT = "/mnt/sda/zf/landform/results/changzhi_mosaic.vrt"
INTER = "/mnt/sda/zf/landform/results/changzhi_inter"
BND = "/mnt/sda/zf/landform/data/changzhi_boundary.geojson"
OUT = "/mnt/sda/zf/landform/results/changzhi_tiled_smooth.parquet"


def vec_tile(sub, subtr, cls_of):
    rows = []
    for geom, val in rasterio.features.shapes(sub.astype(np.int32), mask=sub > 0, connectivity=8, transform=subtr):
        v = int(val); c = cls_of.get(v)
        if not c:
            continue
        g = _shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            rows.append({"parcel_id": v, "class_id": int(c), "geometry": g})
        else:
            for q in getattr(g, "geoms", []):
                if q.geom_type == "Polygon" and not q.is_empty and q.area > 0:
                    rows.append({"parcel_id": v, "class_id": int(c), "geometry": q})
    return rows


def main():
    t0 = time.time()
    def el():
        return "%.0fmin" % ((time.time() - t0) / 60)
    H, W, H4, W4, tr4, crs = pp._read_grid_meta(VRT, DS)
    idmap = np.load(INTER + "/idmap.npy")
    cls_of = pickle.load(open(INTER + "/cls_of.pkl", "rb"))  # 可信:本会话自写 int->int
    parts = []; nt = 0; nfail = 0
    for r0 in range(0, H4, TILE):
        for c0 in range(0, W4, TILE):
            sub = idmap[r0:r0 + TILE, c0:c0 + TILE]
            if not (sub > 0).any():
                continue
            nt += 1
            subtr = tr4 * Affine.translation(c0, r0)
            rawt = gpd.GeoDataFrame(vec_tile(sub, subtr, cls_of), crs=crs)
            if not len(rawt):
                continue
            # ★ 把 parcel_id 当 class_id 喂 smooth 让它原样带过(按位 carry,生产已验证可靠),
            #   避免脆弱质心 sjoin;平滑后 类别 = cls_of[parcel_id](权威映射)。
            hij = gpd.GeoDataFrame({"class_id": rawt["parcel_id"].astype("int64").values},
                                   geometry=rawt.geometry.values, crs=crs)
            try:
                smt = pp.smooth_coverage(hij, tol=TOL, iters=3)   # 块内曲线;class_id 实为 parcel_id
                if "class_id" not in smt.columns or len(smt) == 0:
                    raise RuntimeError("carry-lost")
                smt = smt.rename(columns={"class_id": "parcel_id"})[["parcel_id", "geometry"]]
            except Exception as e:
                nfail += 1
                print("[tiled] tile %d smooth fail %s -> raw" % (nt, repr(e)[:80]), flush=True)
                smt = rawt[["parcel_id", "geometry"]]
            parts.append(smt)
            if nt % 10 == 0:
                print("[tiled] %d tiles (%d fail) (%s)" % (nt, nfail, el()), flush=True)
    allp = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
    allp["parcel_id"] = allp["parcel_id"].astype("int64")
    print("[tiled] %d tiles(%d fail), %d pieces; dissolve by parcel_id... (%s)" % (nt, nfail, len(allp), el()), flush=True)
    diss = allp.dissolve(by="parcel_id").reset_index()
    diss["class_id"] = diss["parcel_id"].map(lambda p: cls_of.get(int(p), 0)).astype(int)
    diss = diss[diss["class_id"] > 0].copy()
    print("[tiled] dissolved -> %d parcels, 类别取自 cls_of (%s)" % (len(diss), el()), flush=True)
    bgeom = pp.load_boundary(BND, UTM)
    clipped = pp.clip_to_boundary(diss, bgeom, utm=UTM)
    final, rep = postproc.run_postproc(clipped, CLASSES, boundary=bgeom, utm=UTM, skip_gaps=True)
    final.to_parquet(OUT)
    cu = final.to_crs(UTM)
    print("[tiled] DONE %d polys, %.1f km2 (市界14152.6) -> %s (%s)" % (len(final), cu.area.sum() / 1e6, OUT, el()), flush=True)
    print("TILED_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
