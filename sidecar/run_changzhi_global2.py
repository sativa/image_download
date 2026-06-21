"""run_changzhi_global2 — 干净全局长治(prefecture 尺度避开 topojson OOM)。
全局 idmap 是一个分区 → vectorize_idmap + coverage_simplify 已是无缝零重叠成品;
topojson+Chaikin(只做曲线磨边)在长治全市路网巨斑下 OOM(>503GB),故**跳过**,改 coverage_simplify-only。
memmap 4 卡推理(已并入)。存 idmap.npy + cls_of.pkl(完整 de-risk)。
注:执行体必须在 if __name__=='__main__' 守卫内(memmap 用 spawn,否则子进程重跑顶层 -> fork 炸)。
"""
import sys, time, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import shapely
import geopandas as gpd
import parcel_pipeline as pp
import postproc

CLASSES = [(1, "耕地", "cropland", (60, 180, 75)), (2, "园地", "orchard", (170, 255, 90)),
           (3, "林地", "forest", (0, 100, 0)), (4, "草地", "grassland", (190, 220, 100)),
           (5, "水体", "water", (0, 130, 200)), (6, "建筑", "built-up", (230, 25, 75)),
           (7, "荒漠", "bare", (170, 140, 100))]
UTM = "EPSG:32649"; TOL = 15.0; DS = 4
MOSAIC = "/mnt/sda/zf/landform/results/changzhi_mosaic.vrt"
WEIGHTS = "/mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt"
BACKBONE = "/home/ps/landform/dinov3/dinov3-vitl16-sat493m"
BND = "/mnt/sda/zf/landform/data/changzhi_boundary.geojson"
INTER = "/mnt/sda/zf/landform/results/changzhi_inter"
OUT = "/mnt/sda/zf/landform/results/changzhi_FINAL.parquet"


def main():
    t0 = time.time()
    def el():
        return "%.0fmin" % ((time.time() - t0) / 60)

    cls, dist, bnd, tr4, crs, cov = pp.infer_global_memmap(MOSAIC, WEIGHTS, BACKBONE, DS, ["0", "1", "2", "3"])
    print("[g2] infer done cov=%.4f (%s)" % (cov, el()), flush=True)
    idmap, cls_of = pp.idmap_from_heads(cls, dist, bnd, ridge=True)
    del cls, dist, bnd
    np.save(INTER + "/idmap.npy", idmap.astype(np.int32))
    pickle.dump(cls_of, open(INTER + "/cls_of.pkl", "wb"))   # 也存 cls_of -> 后处理可从中间重起不重推理
    print("[g2] idmap %d instances saved (%s)" % (len(cls_of), el()), flush=True)
    raw = pp.vectorize_idmap(idmap, cls_of, tr4, crs)
    del idmap
    print("[g2] vectorized %d polys (%s)" % (len(raw), el()), flush=True)
    # coverage_simplify ONLY(拓扑保持去 /4 阶梯;跳过 topojson+Chaikin 避 OOM)
    raw["geometry"] = shapely.coverage_simplify(raw.geometry.values, TOL)
    raw = raw[~raw.geometry.is_empty & raw.geometry.notna()]
    print("[g2] coverage_simplify(tol=%.0f) done (%s)" % (TOL, el()), flush=True)
    bgeom = pp.load_boundary(BND, UTM)
    clipped = pp.clip_to_boundary(raw, bgeom, utm=UTM)
    print("[g2] clipped %d polys (%s)" % (len(clipped), el()), flush=True)
    final, rep = postproc.run_postproc(clipped, CLASSES, boundary=bgeom, utm=UTM, skip_gaps=True)
    final.to_parquet(OUT)
    cu = final.to_crs(UTM)
    print("[g2] DONE %d polys, total %.1f km2 -> %s (%s)" % (len(final), cu.area.sum() / 1e6, OUT, el()), flush=True)
    ar = cu.assign(_l=final["label"].values).groupby("_l").apply(lambda d: round(d.geometry.area.sum() / 1e6, 1))
    print("[g2] per-class km2:", dict(ar), flush=True)
    print("G2_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
