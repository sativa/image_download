"""榆中终版 v2b = 修好的 giant-skip(只跳线状路网, 不误伤草地/林地)smooth_coverage(iters=3)
   -> 裁县界620123 -> postproc.run_postproc(skip_gaps) -> yuzhong_FINAL_v2b.parquet.

与 v2 的唯一差别: _giant_adjacent_arcs 现要求 giant 且 线状(shape_ratio>=10万)才跳 Chaikin,
草地/林地紧凑大团块的田块边重新照常 Chaikin 平滑。**强制重平滑**(不复用旧判据的中间品)。
免重推理: 输入仍是已有原始 shapes coverage yuzhong_global_region.parquet(Chaikin 之前)。
"""
import sys, time, json, os
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely import make_valid

import parcel_pipeline as pp
import postproc
from dino_parcel_export import CLASSES

RES = "/mnt/sda/zf/landform/results"
REGION = f"{RES}/yuzhong_global_region.parquet"
CB = "/tmp/yz_county_boundary.parquet"
DLTB = "/home/ps/landform/data/v11_dltb/620123.parquet"
OUT = f"{RES}/yuzhong_FINAL_v2b.parquet"
SMINT = f"{RES}/yz_v2b_clipped_intermediate.parquet"   # 新路径, 不撞 v2 旧判据中间品
UTM = "EPSG:32648"
M = {"01": "耕地", "02": "园地", "03": "林地", "04": "草地", "05": "建筑", "06": "建筑",
     "07": "建筑", "08": "建筑", "09": "建筑", "10": "建筑", "11": "水体", "12": "荒漠"}
CLS_ORDER = ["耕地", "园地", "林地", "草地", "建筑", "水体", "荒漠"]


def sliver_counts(geoms):
    a = shapely.area(geoms); p = shapely.length(geoms)
    w = np.where(p > 0, a / np.maximum(p, 1e-9), 0)
    out = {}
    for wt in [0.5, 1.0, 1.5, 2.0]:
        out["w<%.1f&a<100" % wt] = int(((w < wt) & (a < 100) & (p > 0) & (a > 0)).sum())
    out["degenerate"] = int(((a <= 0) | (p <= 0)).sum())
    return out


def area_recon(out):
    dl = gpd.read_parquet(DLTB).to_crs(UTM)
    dl["k"] = dl["DLBM"].astype(str).str[:2].map(M).fillna("荒漠")
    D = dl.assign(aa=dl.geometry.area / 1e6).groupby("k")["aa"].sum()
    P = out.to_crs(UTM).assign(aa=lambda d: d.geometry.area / 1e6).groupby("label")["aa"].sum()
    T = pd.DataFrame({"TRUTH": D, "PRED": P}).reindex(CLS_ORDER).fillna(0)
    return T


def main():
    t0 = time.time()
    g = gpd.read_parquet(REGION)
    g = g[["class_id", "geometry"]].copy()
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    vc0 = shapely.get_num_coordinates(g.to_crs("EPSG:3857").geometry.values)
    print("[v2b] loaded REGION %d polys | rawV max=%d (%.0fs)" %
          (len(g), int(vc0.max()), time.time() - t0), flush=True)

    cb = gpd.read_parquet(CB).to_crs(UTM)
    cbgeo = make_valid(shapely.union_all(cb.geometry.values))

    # 强制重平滑(新判据): 不复用旧中间品
    if os.path.exists(SMINT):
        os.remove(SMINT)
    smoothed = pp.smooth_coverage(g, tol=5.0, iters=3, work_crs="EPSG:3857",
                                  giant_vertex_thr=50000, linear_shape_ratio_thr=100000.0)
    vc_sm = shapely.get_num_coordinates(smoothed.geometry.values)
    print("[v2b] smoothed %d polys | smoothedV max=%d mean=%.1f (%.0fs)" %
          (len(smoothed), int(vc_sm.max()), vc_sm.mean(), time.time() - t0), flush=True)
    clipped = pp.clip_to_boundary(smoothed, cbgeo, utm=UTM)
    clipped.to_parquet(SMINT)
    print("[v2b] clipped to 620123 -> %d polys, saved intermediate (%.0fs)" %
          (len(clipped), time.time() - t0), flush=True)

    # 标准强化后处理(无缝 coverage -> skip_gaps)
    classes = [[c[0], c[1], c[2], list(c[3])] for c in CLASSES]
    final, report = postproc.run_postproc(clipped, classes, boundary=cbgeo, utm=UTM, skip_gaps=True)
    inv = ~final.geometry.is_valid
    if bool(inv.any()):
        print("[v2b] final make_valid on %d invalid" % int(inv.sum()), flush=True)
        final.loc[inv, "geometry"] = final.loc[inv, "geometry"].apply(make_valid)
        final = final[final.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        final = final[final.geometry.is_valid & ~final.geometry.is_empty].reset_index(drop=True)
    final.to_parquet(OUT)
    print("[v2b] SAVED %s | %d polys (%.0fs)" % (OUT, len(final), time.time() - t0), flush=True)

    # ===== 验收 =====
    fu = final.to_crs(UTM)
    geoms = fu.geometry.values
    allv = bool(fu.is_valid.all())
    vc = shapely.get_num_coordinates(geoms)
    bm = fu["label"].values == "建筑"
    build_maxV = int(vc[bm].max()) if bm.any() else 0
    gm = fu["label"].values == "草地"
    grass_maxV = int(vc[gm].max()) if gm.any() else 0
    sl = sliver_counts(geoms)
    a_sum = shapely.area(geoms).sum()
    tree = shapely.STRtree(geoms)
    pairs = tree.query(geoms, predicate="overlaps")
    seen = set(); ov_area = 0.0
    for i, j in zip(pairs[0], pairs[1]):
        i, j = int(i), int(j)
        if i >= j or (i, j) in seen:
            continue
        seen.add((i, j))
        try:
            ov_area += geoms[i].intersection(geoms[j]).area
        except Exception:
            pass
    u = shapely.union_all(geoms)
    gap = cbgeo.difference(u).area
    T = area_recon(final)
    T["dpp"] = ((T["PRED"] / max(T["PRED"].sum(), 1e-9) - T["TRUTH"] / max(T["TRUTH"].sum(), 1e-9)) * 100).round(2)
    pred_tot = float(T["PRED"].sum()); truth_tot = float(T["TRUTH"].sum())

    print("\n==================== 榆中 FINAL_v2b 验收 ====================", flush=True)
    print("n_polys          : %d" % len(final), flush=True)
    print("all_valid        : %s" % allv, flush=True)
    print("建筑类 maxV       : %d  (v2 = 690813, 旧SMOOTH3 = 5113760)" % build_maxV, flush=True)
    print("草地类 maxV       : %d  (应较 v2 上升=重新平滑成曲线)" % grass_maxV, flush=True)
    print("verts mean/median: %.1f / %d" % (vc.mean(), int(np.median(vc))), flush=True)
    print("sliver           : %s" % json.dumps(sl, ensure_ascii=False), flush=True)
    print("零重叠 overlap_m2 : %.4f  (a_sum=%.4f km2, pct=%.8f%%)" %
          (ov_area, a_sum / 1e6, ov_area / a_sum * 100 if a_sum else 0), flush=True)
    print("无空白 gap_m2     : %.2f  (%.6f%% of county)" % (gap, gap / cbgeo.area * 100), flush=True)
    print("\n面积守恒 vs DLTB 三调 (km2):", flush=True)
    print(T.round(2).to_string(), flush=True)
    print("PRED total %.1f vs TRUTH %.1f km2 (%+.3f%%)" %
          (pred_tot, truth_tot, (pred_tot / truth_tot * 100 - 100) if truth_tot else 0), flush=True)
    print("================================================================\n", flush=True)

    rec = {"path": OUT, "n_polys": int(len(final)), "all_valid": allv,
           "build_maxV": build_maxV, "build_maxV_v2": 690813, "build_maxV_old": 5113760,
           "grass_maxV": grass_maxV,
           "verts_mean": float(vc.mean()), "verts_median": int(np.median(vc)),
           "slivers": sl, "overlap_m2": float(ov_area), "overlap_pct": float(ov_area / a_sum * 100 if a_sum else 0),
           "gap_m2": float(gap), "gap_pct": float(gap / cbgeo.area * 100),
           "pred_total_km2": pred_tot, "truth_total_km2": truth_tot,
           "area_recon": {k: {"TRUTH": float(T.loc[k, "TRUTH"]), "PRED": float(T.loc[k, "PRED"]),
                              "dpp": float(T.loc[k, "dpp"])} for k in T.index}}
    with open(f"{RES}/yz_v2b_SAVED.json", "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
    print("[v2b] DONE (%.0fmin)" % ((time.time() - t0) / 60), flush=True)
    print("V2B_OK", flush=True)


if __name__ == "__main__":
    main()
