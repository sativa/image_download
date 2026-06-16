"""诊断: 榆中原始 shapes coverage 在 coverage_simplify(tol=5) 后, 列出所有"巨型图斑"(顶点>=阈值)的
顶点数 / 洞数(interior ring) / 形状比(perimeter^2/area), 看建筑路网(线状网络)与草地/林地(紧凑团块)
能否用 洞数 或 形状比 类无关地分开。只诊断, 不写产物。"""
import sys, time
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import geopandas as gpd
import shapely
from shapely import make_valid, coverage_simplify

REGION = "/mnt/sda/zf/landform/results/yuzhong_global_region.parquet"
THR = 50000     # 顶点阈值(现行 giant 判据)


def n_holes(geom):
    if geom is None:
        return 0
    gt = geom.geom_type
    if gt == "Polygon":
        return len(geom.interiors)
    if gt == "MultiPolygon":
        return sum(len(p.interiors) for p in geom.geoms)
    return 0


def main():
    t0 = time.time()
    g = gpd.read_parquet(REGION)[["class_id", "geometry"]].copy()
    g = g.to_crs("EPSG:3857").reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    print("[diag] loaded %d polys (%.0fs)" % (len(g), time.time() - t0), flush=True)

    simp = coverage_simplify(g.geometry.values, tolerance=5.0, simplify_boundary=True)
    simp = np.array([sg if (sg is not None and sg.is_valid) else make_valid(sg) for sg in simp], dtype=object)
    print("[diag] coverage_simplify done (%.0fs)" % (time.time() - t0), flush=True)

    cid = g["class_id"].values
    CLSN = {1: "耕地", 2: "园地", 3: "林地", 4: "草地", 5: "水体", 6: "建筑", 7: "荒漠"}
    nv = shapely.get_num_coordinates(simp)
    area = shapely.area(simp)
    per = shapely.length(simp)
    holes = np.array([n_holes(s) for s in simp])
    shape_ratio = np.where(area > 0, per * per / np.maximum(area, 1e-9), 0)

    giants = np.where(nv >= THR)[0]
    print("\n[diag] %d giant polys (verts>=%d):" % (len(giants), THR), flush=True)
    print("%-6s %-6s %10s %8s %12s %12s" % ("idx", "class", "verts", "holes", "shape_r", "area_km2"), flush=True)
    rows = []
    for i in sorted(giants, key=lambda k: -nv[k]):
        cn = CLSN.get(int(cid[i]), "?")
        print("%-6d %-6s %10d %8d %12.1f %12.3f" %
              (i, cn, nv[i], holes[i], shape_ratio[i], area[i] / 1e6), flush=True)
        rows.append((int(cid[i]), cn, int(nv[i]), int(holes[i]), float(shape_ratio[i]), float(area[i] / 1e6)))

    # 参数扫描: 在各候选判据下, 哪些 giant 被判"线状网络"
    print("\n[diag] === 线状判据扫描(对每个 giant, 各判据是否命中) ===", flush=True)
    print("目标: 建筑(路网)命中, 草地/林地团块不命中。", flush=True)
    for hthr in [50, 100, 200, 500, 1000]:
        hit = [CLSN.get(int(cid[i]), "?") for i in giants if holes[i] >= hthr]
        from collections import Counter
        print("  洞数>=%-5d 命中: %s" % (hthr, dict(Counter(hit))), flush=True)
    for sthr in [200, 500, 1000, 2000, 5000, 10000]:
        from collections import Counter
        hit = [CLSN.get(int(cid[i]), "?") for i in giants if shape_ratio[i] >= sthr]
        print("  形状比>=%-6d 命中: %s" % (sthr, dict(Counter(hit))), flush=True)

    # 也看每类的 giant 们的 holes/shape_ratio 分布范围
    print("\n[diag] === 每类 giant 的 洞数/形状比 范围 ===", flush=True)
    from collections import defaultdict
    byc = defaultdict(list)
    for i in giants:
        byc[CLSN.get(int(cid[i]), "?")].append((holes[i], shape_ratio[i], nv[i]))
    for cn, vals in byc.items():
        hs = [v[0] for v in vals]; srs = [v[1] for v in vals]; vs = [v[2] for v in vals]
        print("  %-4s n=%d | holes[min=%d max=%d] shape_r[min=%.0f max=%.0f] verts[max=%d]" %
              (cn, len(vals), min(hs), max(hs), min(srs), max(srs), max(vs)), flush=True)
    print("[diag] DONE (%.0fs)" % (time.time() - t0), flush=True)
    print("DIAG_OK", flush=True)


if __name__ == "__main__":
    main()
