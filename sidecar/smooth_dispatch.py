"""smooth_dispatch — 按规模自动选 全局平滑 vs 分块平滑(两者都出曲线、拓扑保持)。

依据(长治实测,2026-06-21):
  - 全局一次 topojson+Chaikin:榆中 ~10万地块 fit(~20GB);长治 ~29万地块 **OOM(>503GB)**——
    路网路口的弧数墙,提高 coverage_simplify 容差(tol=45 实测)也救不了。
  - topojson 内存随地块数近 **立方** 增长(连通路网),故阈值取保守值。
  - 分块 per-tile smooth + 按全局 instance-id dissolve:每块县级口径 topojson 必 fit,
    dissolve 无缝重建(PROV 实测 实例数Δ0/面积Δ0.04%),可扩省级;代价 ~−1% 面积 + 极少块边 sliver。

阈值:**地块数(= len(cls_of))**。默认 150,000(榆中~10万 fit / 长治~29万 OOM 之间,偏保守;
  黄土高原密度下 ≈ ~5000 km²)。可用 max_global_parcels 覆盖。

本模块 **additive,不改 parcel_pipeline.py**。future workflow 把
  raw = pp.vectorize_idmap(idmap, cls_of, tr4, crs); sm = pp.smooth_coverage(raw, tol, iters)
换成
  sm = smooth_dispatch.smooth_auto(idmap, cls_of, tr4, crs, tol=tol, iters=iters)
即可自动分流(返回同格式:class_id + geometry,work_crs=源 crs)。
"""
import time
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio.features
from shapely.geometry import shape as _shape
from affine import Affine
import parcel_pipeline as pp

DEFAULT_MAX_GLOBAL_PARCELS = 150_000   # > 此值转分块(榆中~10万 fit / 长治~29万 OOM;内存 ~∝ parcels^3)
DEFAULT_TILE_PX = 4000                 # /4 grid ~19km/块,县级口径 topojson 必 fit


def _vec_tile(sub, subtr, cls_of):
    """块内 idmap -> [{parcel_id, geometry}](拆 MultiPolygon,弃背景/小屑)。"""
    rows = []
    for geom, val in rasterio.features.shapes(sub.astype(np.int32), mask=sub > 0,
                                              connectivity=8, transform=subtr):
        v = int(val)
        c = cls_of.get(v)
        if not c:
            continue
        g = _shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            rows.append({"parcel_id": v, "geometry": g})
        else:
            for q in getattr(g, "geoms", []):
                if q.geom_type == "Polygon" and not q.is_empty and q.area > 0:
                    rows.append({"parcel_id": v, "geometry": q})
    return rows


def tiled_smooth_from_idmap(idmap, cls_of, tr4, crs, tol=15.0, iters=3,
                            tile_px=DEFAULT_TILE_PX, verbose=True):
    """分块 per-tile smooth + 按全局 ID dissolve -> 曲线 gdf(class_id+geometry, work_crs=源 crs)。
    parcel_id 借 smooth 的 class_id 槽按位 carry(避开脆弱质心 sjoin),平滑后类别取自 cls_of(权威)。"""
    t0 = time.time()
    H4, W4 = idmap.shape
    parts = []
    nt = 0
    nfail = 0
    for r0 in range(0, H4, tile_px):
        for c0 in range(0, W4, tile_px):
            sub = idmap[r0:r0 + tile_px, c0:c0 + tile_px]
            if not (sub > 0).any():
                continue
            nt += 1
            subtr = tr4 * Affine.translation(c0, r0)
            rows = _vec_tile(sub, subtr, cls_of)
            if not rows:
                continue
            rawt = gpd.GeoDataFrame(rows, crs=crs)
            hij = gpd.GeoDataFrame({"class_id": rawt["parcel_id"].astype("int64").values},
                                   geometry=rawt.geometry.values, crs=crs)
            try:
                smt = pp.smooth_coverage(hij, tol=tol, iters=iters)   # class_id 实为 parcel_id
                if "class_id" not in smt.columns or len(smt) == 0:
                    raise RuntimeError("carry-lost")
                smt = smt.rename(columns={"class_id": "parcel_id"})[["parcel_id", "geometry"]]
            except Exception as e:
                nfail += 1
                if verbose:
                    print("[tiled-smooth] tile %d fail %s -> raw" % (nt, repr(e)[:80]), flush=True)
                smt = rawt[["parcel_id", "geometry"]]
            parts.append(smt)
    allp = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
    allp["parcel_id"] = allp["parcel_id"].astype("int64")
    diss = allp.dissolve(by="parcel_id").reset_index()
    diss["class_id"] = diss["parcel_id"].map(lambda p: cls_of.get(int(p), 0)).astype(int)
    diss = diss[diss["class_id"] > 0][["class_id", "geometry"]].copy()
    if verbose:
        print("[tiled-smooth] %d tiles(%d fail) -> %d parcels (%.0fmin)"
              % (nt, nfail, len(diss), (time.time() - t0) / 60), flush=True)
    return diss


def smooth_auto(idmap, cls_of, tr4, crs, tol=15.0, iters=3,
                max_global_parcels=DEFAULT_MAX_GLOBAL_PARCELS,
                tile_px=DEFAULT_TILE_PX, verbose=True):
    """自动分流:地块数 ≤ 阈值 -> 全局 smooth_coverage(最干净/面积最准);否则 -> 分块 smooth(可扩省级)。
    返回曲线 gdf(class_id + geometry, work_crs=源 crs),与 pp.smooth_coverage 输出同格式。"""
    n_parcels = len(cls_of)
    if n_parcels <= max_global_parcels:
        if verbose:
            print("[smooth_auto] %d 地块 ≤ %d -> 全局 smooth_coverage(曲线最干净)"
                  % (n_parcels, max_global_parcels), flush=True)
        raw = pp.vectorize_idmap(idmap, cls_of, tr4, crs)
        return pp.smooth_coverage(raw, tol=tol, iters=iters)
    if verbose:
        print("[smooth_auto] %d 地块 > %d -> 分块 smooth+dissolve(可扩省级,~−1%%面积)"
              % (n_parcels, max_global_parcels), flush=True)
    return tiled_smooth_from_idmap(idmap, cls_of, tr4, crs, tol=tol, iters=iters,
                                   tile_px=tile_px, verbose=verbose)


if __name__ == "__main__":
    print("smooth_dispatch: DEFAULT_MAX_GLOBAL_PARCELS=%d, DEFAULT_TILE_PX=%d"
          % (DEFAULT_MAX_GLOBAL_PARCELS, DEFAULT_TILE_PX))
    print("用法: sm = smooth_auto(idmap, cls_of, tr4, crs, tol=15, iters=3)")
