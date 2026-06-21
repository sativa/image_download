"""Q1 测试:巨斑(建筑路网)分离 + 字段类 topojson+Chaikin —— 看 prefecture 能否拿到曲线边。
从保存的 idmap.npy + cls_of.pkl(无需重推理)。核心问题:**去掉建筑类后,字段 topojson 还 OOM 吗?**
若字段 smooth 成功 → 巨斑确是内存元凶、方案可行,产出曲线版 changzhi_curved.parquet。
"""
import sys, time, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import parcel_pipeline as pp
import postproc

CLASSES = [(1, "耕地", "cropland", (60, 180, 75)), (2, "园地", "orchard", (170, 255, 90)),
           (3, "林地", "forest", (0, 100, 0)), (4, "草地", "grassland", (190, 220, 100)),
           (5, "水体", "water", (0, 130, 200)), (6, "建筑", "built-up", (230, 25, 75)),
           (7, "荒漠", "bare", (170, 140, 100))]
UTM = "EPSG:32649"; TOL = 15.0; DS = 4
VRT = "/mnt/sda/zf/landform/results/changzhi_mosaic.vrt"
INTER = "/mnt/sda/zf/landform/results/changzhi_inter"
BND = "/mnt/sda/zf/landform/data/changzhi_boundary.geojson"
OUT = "/mnt/sda/zf/landform/results/changzhi_curved.parquet"


def main():
    t0 = time.time()
    def el():
        return "%.0fmin" % ((time.time() - t0) / 60)
    H, W, H4, W4, tr4, crs = pp._read_grid_meta(VRT, DS)
    idmap = np.load(INTER + "/idmap.npy")
    cls_of = pickle.load(open(INTER + "/cls_of.pkl", "rb"))  # 可信:本会话 G2 run 自写的 int->int 字典,非外来
    raw = pp.vectorize_idmap(idmap, cls_of, tr4, crs)
    del idmap
    print("[q1] vectorized %d polys (%s)" % (len(raw), el()), flush=True)
    fields = raw[raw["class_id"] != 6].copy()
    giant = raw[raw["class_id"] == 6].copy()
    print("[q1] fields=%d building=%d (%s)" % (len(fields), len(giant), el()), flush=True)
    # ★ 核心测试:字段类 topojson+Chaikin 是否 fit
    try:
        sm_fields = pp.smooth_coverage(fields, tol=TOL, iters=3)
        print("[q1] ★ FIELDS smooth_coverage(topojson+Chaikin) FIT OK: %d polys (%s)" % (len(sm_fields), el()), flush=True)
    except Exception as e:
        print("[q1] ★ FIELDS smooth FAILED: %s (%s)" % (repr(e)[:200], el()), flush=True)
        raise
    # 建筑:仅 coverage_simplify(路网直边,不需曲线)
    giant["geometry"] = shapely.coverage_simplify(giant.geometry.values, TOL)
    merged = gpd.GeoDataFrame(pd.concat([sm_fields[["class_id", "geometry"]], giant[["class_id", "geometry"]]],
                                        ignore_index=True), crs=raw.crs)
    print("[q1] merged %d polys; resolve_overlaps(字段↔建筑边界)... (%s)" % (len(merged), el()), flush=True)
    merged = pp.resolve_overlaps(merged)
    print("[q1] resolve_overlaps done %d (%s)" % (len(merged), el()), flush=True)
    bgeom = pp.load_boundary(BND, UTM)
    clipped = pp.clip_to_boundary(merged, bgeom, utm=UTM)
    final, rep = postproc.run_postproc(clipped, CLASSES, boundary=bgeom, utm=UTM, skip_gaps=True)
    final.to_parquet(OUT)
    cu = final.to_crs(UTM)
    print("[q1] DONE %d polys, %.1f km2 (市界14152.6) -> %s (%s)" % (len(final), cu.area.sum() / 1e6, OUT, el()), flush=True)
    print("Q1_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
