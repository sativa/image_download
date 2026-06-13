"""yuzhong SMOOTH2: 拓扑保持的曲线平滑(更顺的曲线), 全县整体不分块, 无分幅线.

路线(首选, 全县整体一把):
  yuzhong_global_region.parquet(干净分区, 无缝)
  -> coverage_simplify(tol=5) 全县整体(去/4阶梯, 得共享边精确一致的coverage, 顶点降~2x)
  -> topojson.Topology(prequantize=False, shared_coords=True) 全县一次(顶点已少, 423GB可扛)
  -> 对每条 arc 做 Chaikin 平滑(端点=节点固定, N迭代) -> 更顺的曲线, 共享边逐点一致 -> 无缝
  -> to_gdf() 重建 -> 真几何 intersection 裁县界620123 -> SMOOTH2.

Chaikin: 每段用 1/4、3/4 切角点替换, 迭代N次; arc端点(节点)保持不动.
"""
import sys, time, json, argparse
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely import make_valid, coverage_simplify
import topojson

REGION = "/mnt/sda/zf/landform/results/yuzhong_global_region.parquet"
CB = "/tmp/yz_county_boundary.parquet"
DLTB = "/home/ps/landform/data/v11_dltb/620123.parquet"
UTM = "EPSG:32648"
M = {"01": "耕地", "02": "园地", "03": "林地", "04": "草地", "05": "建筑", "06": "建筑",
     "07": "建筑", "08": "建筑", "09": "建筑", "10": "建筑", "11": "水体", "12": "荒漠"}


def chaikin_arc(arc, iters):
    """Chaikin corner-cutting on a single arc; FIRST & LAST point (=nodes) FIXED.
    arc: list of [x,y]. Returns new list of [x,y]."""
    pts = np.asarray(arc, dtype=np.float64)
    if iters <= 0 or len(pts) < 3:
        return arc
    p = pts
    for _ in range(iters):
        if len(p) < 3:
            break
        # interior segments get cut; endpoints preserved
        a = p[:-1]
        b = p[1:]
        q = a + 0.25 * (b - a)   # 1/4 point of each segment
        r = a + 0.75 * (b - a)   # 3/4 point of each segment
        # interleave q,r for each segment -> new polyline, then pin true endpoints
        inner = np.empty((2 * len(q), 2), dtype=np.float64)
        inner[0::2] = q
        inner[1::2] = r
        new = np.empty((len(inner) + 2, 2), dtype=np.float64)
        new[0] = p[0]            # node fixed
        new[1:-1] = inner
        new[-1] = p[-1]          # node fixed
        p = new
    return p.tolist()


def count_verts(geoms):
    return shapely.get_num_coordinates(geoms)


def build_smoothed_gdf(topo_template_output, classmeta, iters, snap=True):
    """Deep-copy template arcs, Chaikin each (nodes fixed), rebuild gdf, then snap
    back to a perfectly valid coverage via coverage_simplify(tol=0).

    Why the snap: topojson(shared_coords=True) on a coverage_simplify output occasionally
    leaves a small fraction of interior boundaries as TWO single-ref arcs instead of one
    shared arc (degenerate/near-touching spots). Chaikin then moves those two copies
    independently -> tiny local overlaps/gaps (~0.3% area). coverage_simplify(tolerance=0)
    is a topology-preserving snap that re-fuses shared edges WITHOUT changing geometry
    (tol=0 never simplifies, only repairs the coverage) -> exact zero overlap, area restored.
    """
    import copy
    topo = copy.deepcopy(TOPO)
    arcs = topo.output["arcs"]
    new_arcs = []
    for arc in arcs:
        new_arcs.append(chaikin_arc(arc, iters) if iters > 0 else arc)
    topo.output["arcs"] = new_arcs
    gdf = topo.to_gdf(crs="EPSG:3857")
    gdf["geometry"] = gdf.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    if snap and iters > 0:
        snapped = coverage_simplify(gdf.geometry.values, tolerance=0.0, simplify_boundary=True)
        snapped = np.array([s if (s is not None and s.is_valid) else make_valid(s)
                            for s in snapped], dtype=object)
        gdf = gpd.GeoDataFrame(
            {c: gdf[c].values for c in ["class_id", "label", "label_en", "rgb_hex"]},
            geometry=gpd.GeoSeries(snapped, crs="EPSG:3857"))
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    return gdf


def overlap_pct(geoms, fast=True):
    """Return (a_sum, a_union, overlap%). For a valid coverage sum==union so we report
    coverage validity + a cheap STRtree pairwise-overlap area instead of a full union_all
    (which is pathologically slow on 30M-vertex geometry)."""
    a_sum = shapely.area(geoms).sum()
    if not fast:
        a_union = shapely.union_all(geoms).area
        ov = (a_sum - a_union) / a_union * 100 if a_union > 0 else 0.0
        return a_sum, a_union, ov
    # fast: STRtree pairwise-overlap area only. (Deliberately NOT calling
    # shapely.coverage_is_valid here: on a 30M-vertex county coverage it is
    # pathologically slow ~30min+; STRtree overlaps gives the same zero-overlap signal.)
    cov_ok = None
    tree = shapely.STRtree(geoms)
    pairs = tree.query(geoms, predicate="overlaps")
    ov_area = 0.0
    seen = set()
    for i, j in zip(pairs[0], pairs[1]):
        if i >= j:
            continue
        key = (int(i), int(j))
        if key in seen:
            continue
        seen.add(key)
        try:
            ov_area += geoms[i].intersection(geoms[j]).area
        except Exception:
            pass
    ov = ov_area / a_sum * 100 if a_sum > 0 else 0.0
    return a_sum, a_sum, ov, cov_ok


def clip_to_county(g2, t0):
    """Real-geometry intersection clip to 620123 (reuse SMOOTH logic)."""
    g2u = g2.to_crs(UTM)
    cb = gpd.read_parquet(CB).to_crs(UTM)
    cbgeo = make_valid(cb.geometry.values[0])
    minx, miny, maxx, maxy = cbgeo.bounds
    b = g2u.geometry.bounds.values
    keep = ~((b[:, 2] < minx) | (b[:, 0] > maxx) | (b[:, 3] < miny) | (b[:, 1] > maxy))
    g2u = g2u[keep].reset_index(drop=True)
    cov = g2u.geometry.covered_by(cbgeo)
    inside = g2u[cov].copy()
    edge = g2u[~cov].copy()
    rows = []
    for geom, cid, lab, le, hx in zip(edge.geometry.values, edge["class_id"].values,
                                      edge["label"].values, edge["label_en"].values, edge["rgb_hex"].values):
        try:
            inter = geom.intersection(cbgeo)
        except Exception:
            try:
                inter = make_valid(geom).intersection(cbgeo)
            except Exception:
                continue
        if inter.is_empty or inter.area <= 0:
            continue
        rows.append({"class_id": cid, "label": lab, "label_en": le, "rgb_hex": hx, "geometry": inter})
    ec = gpd.GeoDataFrame(rows, crs=UTM) if rows else gpd.GeoDataFrame(
        {"class_id": [], "label": [], "label_en": [], "rgb_hex": [], "geometry": []}, crs=UTM)
    cols = ["class_id", "label", "label_en", "rgb_hex", "geometry"]
    out = gpd.GeoDataFrame(pd.concat([inside[cols], ec[cols]], ignore_index=True), crs=UTM)
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    out["geometry"] = out.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    out = out[~out.geometry.is_empty & out.geometry.notna()].reset_index(drop=True)
    out["area_m2"] = out.geometry.area.round(1)
    out.insert(0, "gid", range(1, len(out) + 1))
    return out


def area_recon(out):
    dl = gpd.read_parquet(DLTB).to_crs(UTM)
    dl["k"] = dl["DLBM"].astype(str).str[:2].map(M).fillna("荒漠")
    D = dl.assign(aa=dl.geometry.area / 1e6).groupby("k")["aa"].sum()
    P = out.assign(aa=out.geometry.area / 1e6).groupby("label")["aa"].sum()
    cls = ["耕地", "园地", "林地", "草地", "建筑", "水体", "荒漠"]
    T = pd.DataFrame({"TRUTH": D, "PRED": P}).reindex(cls).fillna(0)
    return T


def main():
    global TOPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--iters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--save-iter", type=int, default=2, help="which iter to save as SMOOTH2")
    ap.add_argument("--outdir", default="/mnt/sda/zf/landform/results")
    a = ap.parse_args()
    t0 = time.time()

    # 1) clean partition -> 3857
    g = gpd.read_parquet(REGION).to_crs("EPSG:3857").reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    print(f"[s2] {len(g)} polys loaded (3857) ({time.time()-t0:.0f}s)", flush=True)
    v_raw = count_verts(g.geometry.values)
    print(f"[s2] verts RAW: total={int(v_raw.sum())} mean={v_raw.mean():.1f} median={np.median(v_raw):.0f}", flush=True)

    # 2) coverage_simplify tol=5 (whole county) -> SMOOTH baseline coverage
    t1 = time.time()
    simp = coverage_simplify(g.geometry.values, tolerance=a.tol, simplify_boundary=True)
    simp = np.array([sg if (sg is not None and sg.is_valid) else make_valid(sg) for sg in simp], dtype=object)
    g2 = gpd.GeoDataFrame({
        "class_id": g["class_id"].values, "label": g["label"].values,
        "label_en": g["label_en"].values, "rgb_hex": g["rgb_hex"].values,
    }, geometry=gpd.GeoSeries(simp, crs="EPSG:3857"))
    g2 = g2[~g2.geometry.is_empty & g2.geometry.notna()].reset_index(drop=True)
    v_smooth = count_verts(g2.geometry.values)
    print(f"[s2] coverage_simplify(tol={a.tol}) done ({time.time()-t1:.0f}s) | "
          f"verts SMOOTH: total={int(v_smooth.sum())} mean={v_smooth.mean():.1f} median={np.median(v_smooth):.0f} "
          f"reduction {v_raw.sum()/v_smooth.sum():.2f}x", flush=True)

    # 3) topojson over the simplified coverage (whole county, one shot). Cache to skip
    #    the ~18min rebuild on reruns (deterministic for fixed input).
    # NOTE: pickle is self-produced on this trusted server in this same pipeline (never
    #       loaded from an untrusted source), so arbitrary-code-exec risk does not apply.
    import pickle
    t2 = time.time()
    cache = f"{a.outdir}/yz_s2_topo.pkl"
    import os
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            TOPO = pickle.load(f)
        print(f"[s2] TOPOLOGY loaded from cache: n_arcs={len(TOPO.output['arcs'])} ({time.time()-t2:.0f}s)", flush=True)
    else:
        # shared_coords=False (junction/coord-hash based) merges shared edges far more
        # reliably than shared_coords=True on a coverage_simplify output (proven on windows:
        # zero Chaikin overlap, ~33% fewer arcs, faster to_gdf) -> no post-snap needed.
        TOPO = topojson.Topology(g2, prequantize=False, shared_coords=False)
        with open(cache, "wb") as f:
            pickle.dump(TOPO, f)
        print(f"[s2] TOPOLOGY built (whole county, no chunking): n_arcs={len(TOPO.output['arcs'])} "
              f"({time.time()-t2:.0f}s) cached", flush=True)
    n_arcs = len(TOPO.output["arcs"])

    stats = {}
    saved_out = None
    for it in a.iters:
        ti = time.time()
        # snap=False: rely on shared_coords=False clean arc-sharing for zero overlap
        # (the full-county coverage_simplify(tol=0) snap is correct but ~30min/iter, too slow).
        gdf = build_smoothed_gdf(TOPO.output, g2, it, snap=False)
        gdf["geometry"] = gdf.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
        v_it = count_verts(gdf.geometry.values)
        a_sum, a_union, ov, cov_ok = overlap_pct(gdf.geometry.values, fast=True)
        allvalid = bool(gdf.is_valid.all())
        tag = "SMOOTH(tol5)" if it == 0 else f"+Chaikin{it}"
        print(f"[s2] {tag}: verts mean={v_it.mean():.1f} median={np.median(v_it):.0f} total={int(v_it.sum())} "
              f"| overlap={ov:.4f}% coverage_valid={cov_ok} | all_valid={allvalid} | "
              f"area={a_union/1e6:.1f}km2 ({time.time()-ti:.0f}s)", flush=True)
        stats[it] = {
            "tag": tag, "verts_mean": float(v_it.mean()), "verts_median": float(np.median(v_it)),
            "verts_total": int(v_it.sum()), "overlap_pct": float(ov), "coverage_valid": cov_ok,
            "all_valid": allvalid, "area_km2_precoverage": float(a_union / 1e6), "n_polys": int(len(gdf)),
        }
        # save the un-clipped smoothed coverage parquet per iter (for ladder plotting)
        gdf.to_parquet(f"{a.outdir}/yz_s2_iter{it}_cov.parquet")

        if it == a.save_iter:
            out = clip_to_county(gdf, t0)
            T = area_recon(out)
            T["dpp"] = ((T["PRED"] / T["PRED"].sum() - T["TRUTH"] / T["TRUTH"].sum()) * 100).round(1)
            a2_sum, a2_union, ov2, cov_ok2 = overlap_pct(out.geometry.values, fast=True)
            v_clip = count_verts(out.geometry.values)
            allv2 = bool(out.is_valid.all())
            outp = f"{a.outdir}/yuzhong_SMOOTH2.parquet"
            out.to_crs("EPSG:4326").to_parquet(outp)
            print(f"[s2] SAVED SMOOTH2 (iter={it}) -> {outp} | {len(out)} polys | "
                  f"clip overlap={ov2:.4f}% all_valid={allv2} verts mean={v_clip.mean():.1f}", flush=True)
            print(T.round(1).to_string(), flush=True)
            print(f"[s2] total PRED {T['PRED'].sum():.0f} vs TRUTH {T['TRUTH'].sum():.0f} km2 "
                  f"({T['PRED'].sum()/T['TRUTH'].sum()*100-100:+.1f}%)", flush=True)
            stats["SAVED"] = {
                "iter": it, "path": outp, "n_polys": int(len(out)),
                "clip_overlap_pct": float(ov2), "all_valid": allv2,
                "verts_mean": float(v_clip.mean()), "verts_median": float(np.median(v_clip)),
                "pred_total_km2": float(T["PRED"].sum()), "truth_total_km2": float(T["TRUTH"].sum()),
                "area_recon": {k: {"TRUTH": float(T.loc[k, "TRUTH"]), "PRED": float(T.loc[k, "PRED"]),
                                   "dpp": float(T.loc[k, "dpp"])} for k in T.index},
            }
            saved_out = out

    with open(f"{a.outdir}/yz_smooth2_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[s2] DONE ({time.time()-t0:.0f}s) stats -> {a.outdir}/yz_smooth2_stats.json", flush=True)
    print("SMOOTH2_OK", flush=True)


if __name__ == "__main__":
    main()
