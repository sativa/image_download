"""高 tol 一次性平滑测试:能否一次生成曲线版长治(prefecture)。
从保存 idmap.npy+cls_of.pkl(零重推理)。核心:coverage_simplify(tol=45,比失败的15狠3x,拓扑保持)
压低巨斑顶点(+去<45m小洞减弧数)→ 一次 smooth_coverage(=coverage_simplify+topojson+giant-skip Chaikin),
**一个 coverage、不拆不合并**(绕开 Q1 的 resolve_overlaps 合并墙)。topojson 若 fit → 一次拿到曲线无缝长治。
"""
import sys, time, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import parcel_pipeline as pp
import postproc

CLASSES = [(1, "耕地", "cropland", (60, 180, 75)), (2, "园地", "orchard", (170, 255, 90)),
           (3, "林地", "forest", (0, 100, 0)), (4, "草地", "grassland", (190, 220, 100)),
           (5, "水体", "water", (0, 130, 200)), (6, "建筑", "built-up", (230, 25, 75)),
           (7, "荒漠", "bare", (170, 140, 100))]
UTM = "EPSG:32649"; TOL = 45.0; DS = 4
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
    cls_of = pickle.load(open(INTER + "/cls_of.pkl", "rb"))  # 可信:本会话自写 int->int
    raw = pp.vectorize_idmap(idmap, cls_of, tr4, crs)
    del idmap
    print("[hitol] vectorized %d polys (%s)" % (len(raw), el()), flush=True)
    # ★ 一次性平滑:coverage_simplify(45)+topojson+giant-skip Chaikin。测 topojson 是否 fit。
    sm = pp.smooth_coverage(raw, tol=TOL, iters=3)
    print("[hitol] ★ smooth_coverage(tol=%g) FIT OK: %d polys (%s)" % (TOL, len(sm), el()), flush=True)
    bgeom = pp.load_boundary(BND, UTM)
    clipped = pp.clip_to_boundary(sm, bgeom, utm=UTM)
    final, rep = postproc.run_postproc(clipped, CLASSES, boundary=bgeom, utm=UTM, skip_gaps=True)
    final.to_parquet(OUT)
    cu = final.to_crs(UTM)
    print("[hitol] DONE %d polys, %.1f km2 (市界14152.6) -> %s (%s)" % (len(final), cu.area.sum() / 1e6, OUT, el()), flush=True)
    print("HITOL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
