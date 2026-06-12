"""Stage B —— 分块拓扑感知FFL矢量化 + 县界裁剪 + 全县QA(无空白/无空心).
读 Stage A 产物 (/4 idmap, ffc0/ffc2, cls_of, tr4) ->
  按规则网格把 /4 idmap 切成 NBX x NBY 块(块边=raster直切, 共享 cut line);
  每块: shapes(子idmap) -> 块内 coverage gdf -> topo_ffl(ff子窗+子transform, clamp16)
        块边arc(沿块raster边界跑的arc)不做帧场snap, 直接拉直为cut线段 -> 两侧块完全一致 -> 无缝;
  ProcessPool 铺 N 核 -> 合并所有块 -> 真几何 intersection 裁县界620123 ->
  平滑/属性 -> QA(gaps/holes 填补) -> 存 yuzhong_FFL_FINAL.parquet + 预览/放大/QA png。
零重叠由拓扑结构保证(每条共享边只正则一次); 块间无缝由共享cut线保证。
"""
import sys, os, json, time, math
from pathlib import Path
import numpy as np

SIDECAR = "/home/ps/landform/sidecar"
sys.path.insert(0, SIDECAR)
ART = "/mnt/sda/zf/landform/results/yz_ffl_artifacts"
OUTDIR = "/mnt/sda/zf/landform/results"
DLTB = "/home/ps/landform/data/v11_dltb/620123.parquet"
CB = "/tmp/yz_county_boundary.parquet"
OUT = OUTDIR + "/yuzhong_FFL_FINAL.parquet"
CLIP_CKPT = OUTDIR + "/yz_ffl_clipped_ckpt.parquet"   # resume point (post-clip, pre-QA)
UTM = "EPSG:32648"
NBX = 12; NBY = 12                     # 12x12=144 块, ~96369/144 ~ 670 parcels/块
HALO = 0                               # 块边=raster直切(共享cut), 无需halo

HEX = {"耕地": "#3cb44b", "园地": "#aaff5a", "林地": "#006400", "草地": "#bedc64",
       "水体": "#0082c8", "建筑": "#e6194b", "荒漠": "#aa8c64"}
EN = {"耕地": "cropland", "园地": "orchard", "林地": "forest", "草地": "grassland",
      "水体": "water", "建筑": "built", "荒漠": "baresoil"}
CLSID2ZH = None  # filled from dino_parcel_export.NAME_ZH


def block_worker(args):
    """处理一个 /4 idmap 子块, 返回 (geoms_wkb, labels) list."""
    (bi, bj, r0, r1, c0, c1, tr4_6, crs_str) = args
    import cv2
    import rasterio.features
    from rasterio.transform import from_origin
    from affine import Affine
    from shapely.geometry import shape as _shape, LineString
    from shapely import wkb as _wkb
    import geopandas as gpd
    import topojson
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    from dino_parcel_export import NAME_ZH
    from ff_polygonize import ff_main_angle

    idmap = np.load(ART + "/idmap4.npy", mmap_mode="r")
    ffc0_g = np.load(ART + "/ffc0.npy", mmap_mode="r")
    ffc2_g = np.load(ART + "/ffc2.npy", mmap_mode="r")
    cls_of = {int(k): int(v) for k, v in json.load(open(ART + "/cls_of.json")).items()}

    sub = np.array(idmap[r0:r1, c0:c1], dtype=np.int32)
    if not (sub > 0).any():
        return []
    ffc0 = np.array(ffc0_g[r0:r1, c0:c1])
    ffc2 = np.array(ffc2_g[r0:r1, c0:c1])
    H, W = sub.shape
    a = Affine(*tr4_6)
    # 子块 transform: 原点平移到 (c0,r0)
    sub_tr = a * Affine.translation(c0, r0)
    inv = ~sub_tr

    # shapes -> 块内 coverage gdf (parcel_id, class_id)
    rows = []
    for geom, val in rasterio.features.shapes(sub, mask=sub > 0, connectivity=8, transform=sub_tr):
        c = cls_of.get(int(val))
        if not c:
            continue
        g = _shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        rows.append({"parcel_id": int(val), "class_id": c, "geometry": g})
    if not rows:
        return []
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_str)

    # ----- 拓扑FFL: 每条arc正则一次 -----
    topo = topojson.Topology(gdf, prequantize=False, shared_coords=True)
    arcs = topo.output["arcs"]
    # 块raster边界(CRS)用于判断块边arc
    bx0, bx1 = sub_tr * (0, 0), sub_tr * (W, H)   # (lon,lat) at corners
    lon_lo, lon_hi = min(bx0[0], bx1[0]), max(bx0[0], bx1[0])
    lat_lo, lat_hi = min(bx0[1], bx1[1]), max(bx0[1], bx1[1])
    px_lon = abs(a.a); px_lat = abs(a.e)
    tol_lon = px_lon * 1.5; tol_lat = px_lat * 1.5

    def on_edge(x, y):
        return (abs(x - lon_lo) < tol_lon or abs(x - lon_hi) < tol_lon or
                abs(y - lat_lo) < tol_lat or abs(y - lat_hi) < tol_lat)

    new_arcs = []
    for arc in arcs:
        n = len(arc)
        closed = (n >= 2 and abs(arc[0][0] - arc[-1][0]) < 1e-9 and abs(arc[0][1] - arc[-1][1]) < 1e-9)
        edge_frac = sum(1 for (x, y) in arc if on_edge(x, y)) / max(n, 1)
        # 块边arc(沿块raster边界跑): 不做帧场snap, 只做拓扑保持的拉直(消/4阶梯, 两侧块一致 -> 无缝)。
        # 闭合arc(整圈, 如紧贴块边的整块)绝不能拉直成端点; 用闭合DP简化。
        if edge_frac > 0.85 and n >= 2:
            if closed:
                ap2 = np.array(arc, np.float32).reshape(-1, 1, 2)
                sm = cv2.approxPolyDP(ap2, float(tol_lon), True)[:, 0, :].tolist()  # closed DP
                if len(sm) >= 3:
                    sm.append(list(sm[0]))                 # re-close
                    new_arcs.append([list(p) for p in sm]); continue
                new_arcs.append([list(p) for p in arc]); continue
            # 开放块边arc: 近共线则拉直为端点直线(块角真转角则闭合DP不snap)
            (xa, ya), (xb, yb) = arc[0], arc[-1]
            seg = math.hypot(xb - xa, yb - ya)
            if seg < 1e-9:
                new_arcs.append([list(p) for p in arc]); continue
            maxd = max((abs((xb - xa) * (ya - yp) - (xa - xp) * (yb - ya)) / seg) for (xp, yp) in arc[1:-1]) if n > 2 else 0.0
            if maxd < tol_lon * 1.0:
                new_arcs.append([list(arc[0]), list(arc[-1])])   # straight cut, identical from both sides
                continue
            ap2 = np.array(arc, np.float32).reshape(-1, 1, 2)
            sm = cv2.approxPolyDP(ap2, float(tol_lon), False)[:, 0, :].tolist()
            sm[0] = list(arc[0]); sm[-1] = list(arc[-1])
            new_arcs.append([list(p) for p in sm])
            continue
        # 普通内部arc: DP简化 + 帧场snap + clamp16, 端点固定
        px = [inv * (x, y) for (x, y) in arc]
        ap = np.array(px, np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(ap, 2.0, False)[:, 0, :]
        if len(approx) < 2:
            approx = np.array(px, np.float32)
        approx = approx.tolist()
        approx[0] = list(px[0]); approx[-1] = list(px[-1])
        coords = [tuple(p) for p in approx]
        nseg = len(coords) - 1
        if nseg < 2:
            reg_px = coords
        else:
            lines = []
            for i in range(nseg):
                (x0, y0), (x1, y1) = coords[i], coords[i + 1]
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                cy = min(max(int(my), 0), H - 1); cx = min(max(int(mx), 0), W - 1)
                th = ff_main_angle(ffc0[cy, cx], ffc2[cy, cx])
                edge_ang = math.atan2(y1 - y0, x1 - x0)
                cands = [th, th + math.pi / 2]
                best = min(cands, key=lambda cc: abs(((edge_ang - cc + math.pi / 2) % math.pi) - math.pi / 2))
                d = abs(((edge_ang - best + math.pi / 2) % math.pi) - math.pi / 2)
                lines.append((mx, my, best if math.degrees(d) < 35.0 else edge_ang))
            out = [tuple(coords[0])]
            for i in range(1, nseg):
                l1, l2 = lines[i - 1], lines[i]
                x1p, y1p, a1 = l1; x2p, y2p, a2 = l2
                d1 = (math.cos(a1), math.sin(a1)); d2 = (math.cos(a2), math.sin(a2))
                den = d1[0] * d2[1] - d1[1] * d2[0]
                orig = coords[i]
                if abs(den) < 1e-6:
                    p = orig
                else:
                    tt = ((x2p - x1p) * d2[1] - (y2p - y1p) * d2[0]) / den
                    p = (x1p + tt * d1[0], y1p + tt * d1[1])
                    if abs(p[0] - orig[0]) + abs(p[1] - orig[1]) > 16.0:   # clamp16
                        p = orig
                out.append((float(p[0]), float(p[1])))
            out.append(tuple(coords[-1]))
            reg_px = out
        reg_crs = [list(sub_tr * (cx, cy)) for (cx, cy) in reg_px]
        reg_crs[0] = list(arc[0]); reg_crs[-1] = list(arc[-1])
        new_arcs.append(reg_crs)
    topo.output["arcs"] = new_arcs
    outg = topo.to_gdf()
    outg = outg.set_crs(gdf.crs, allow_override=True)
    outg["geometry"] = [g if g.is_valid else g.buffer(0) for g in outg.geometry]
    outg = outg[~outg.geometry.is_empty].reset_index(drop=True)

    res = []
    for g, cid in zip(outg.geometry.values, outg["class_id"].values):
        if g is None or g.is_empty:
            continue
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for p in polys:
            if p.geom_type == "Polygon" and not p.is_empty and p.area > 0:
                res.append((p.wkb, NAME_ZH[int(cid)]))
    return res


def main():
    import geopandas as gpd
    import pandas as pd
    from shapely import wkb as _wkb, make_valid, unary_union
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dino_parcel_export import smooth_geom, NAME_ZH
    t0 = time.time()
    meta = json.load(open(ART + "/meta.json"))
    H4, W4 = meta["H4"], meta["W4"]; tr4_6 = meta["tr4"]; crs_str = meta["crs"]
    print(f"[B] /4 grid {H4}x{W4} crs={crs_str[:24]} tr4={tr4_6}", flush=True)

    cb_g = gpd.read_parquet(CB).to_crs(UTM)
    cbgeo = make_valid(cb_g.geometry.values[0])
    print(f"[B] county boundary {cbgeo.area/1e6:.1f} km2", flush=True)

    if os.path.exists(CLIP_CKPT):
        out = gpd.read_parquet(CLIP_CKPT).to_crs(UTM)
        print(f"[B] RESUME from clip checkpoint: {len(out)} polys ({CLIP_CKPT})", flush=True)
    else:
        # 块网格 (按 raster 行列均分)
        rb = np.linspace(0, H4, NBY + 1).astype(int)
        cb = np.linspace(0, W4, NBX + 1).astype(int)
        jobs = []
        for bi in range(NBY):
            for bj in range(NBX):
                jobs.append((bi, bj, int(rb[bi]), int(rb[bi + 1]), int(cb[bj]), int(cb[bj + 1]), tr4_6, crs_str))
        print(f"[B] {len(jobs)} blocks ({NBX}x{NBY}) -> ProcessPool", flush=True)
        all_geoms = []; all_labels = []
        done = 0
        with ProcessPoolExecutor(max_workers=32) as ex:
            futs = {ex.submit(block_worker, j): (j[0], j[1]) for j in jobs}
            for fu in as_completed(futs):
                res = fu.result(); done += 1
                for wkbb, lab in res:
                    all_geoms.append(_wkb.loads(wkbb)); all_labels.append(lab)
                if done % 24 == 0 or done == len(jobs):
                    print(f"[B]   blocks {done}/{len(jobs)} | parcels so far {len(all_geoms)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[B] topoFFL done: {len(all_geoms)} block-parcels ({time.time()-t0:.0f}s)", flush=True)

        g = gpd.GeoDataFrame({"label": all_labels}, geometry=all_geoms, crs=crs_str).to_crs(UTM)
        g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
        g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)

        # ----- 真几何裁剪到县界620123 (巨型no-data水体先simplify加速intersection) -----
        minx, miny, maxx, maxy = cbgeo.bounds
        bb = g.geometry.bounds.values
        keep = ~((bb[:, 2] < minx) | (bb[:, 0] > maxx) | (bb[:, 3] < miny) | (bb[:, 1] > maxy))
        g = g[keep].reset_index(drop=True)
        cov_in = g.geometry.covered_by(cbgeo)
        inside = g[cov_in].copy(); edgers = g[~cov_in].copy()
        print(f"[B] fully-inside {len(inside)} | boundary {len(edgers)} -> clip ({time.time()-t0:.0f}s)", flush=True)
        clipped = []
        for geom, lab in zip(edgers.geometry.values, edgers["label"].values):
            try:
                inter = geom.intersection(cbgeo)
            except Exception:
                try:
                    inter = make_valid(geom).intersection(cbgeo)
                except Exception:
                    continue
            if inter.is_empty or inter.area <= 0:
                continue
            clipped.append({"label": lab, "geometry": inter})
        ec = gpd.GeoDataFrame(clipped, crs=UTM) if clipped else gpd.GeoDataFrame({"label": [], "geometry": []}, crs=UTM)
        out = gpd.GeoDataFrame(pd.concat([inside[["label", "geometry"]], ec[["label", "geometry"]]], ignore_index=True), crs=UTM)
        out = out.explode(index_parts=False).reset_index(drop=True)
        out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
        print(f"[B] clipped -> {len(out)} polys ({time.time()-t0:.0f}s)", flush=True)
        out.to_parquet(CLIP_CKPT)            # resume checkpoint (skip 22-min clip on rerun)
        print(f"[B] clip checkpoint -> {CLIP_CKPT}", flush=True)

    # ----- QA: 无空白/无空心 (在 union 之前用原始parcel做gap检测, 之后填补) -----
    out["geometry"] = out.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    out = out[~out.geometry.is_empty & out.geometry.notna()].reset_index(drop=True)
    out, qa_gaps, qa_holes = qa_and_fill(out, cbgeo, t0)
    make_qa_fig(out, cbgeo, qa_gaps, qa_holes, t0)

    # ----- 属性 + 存 (不做 Chaikin 平滑: topo-FFL 边已规整且共享边唯一; 逐地块Chaikin会破坏
    #        共享边一致性 -> 重新引入gap/overlap, 故跳过) -----
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    out["label_en"] = out["label"].map(EN); out["rgb_hex"] = out["label"].map(HEX)
    z2i = {v: k for k, v in NAME_ZH.items()}
    out["class_id"] = out["label"].map(z2i)
    out["area_m2"] = out.geometry.area.round(1)
    out.insert(0, "gid", range(1, len(out) + 1))
    out_4326 = out.to_crs("EPSG:4326")
    out_4326 = out_4326[["gid", "class_id", "label", "label_en", "rgb_hex", "area_m2", "geometry"]]
    out_4326.to_parquet(OUT)
    print(f"[B] FINAL {len(out_4326)} polys -> {OUT} ({time.time()-t0:.0f}s)", flush=True)

    # 顶点统计
    nv = vert_stats(list(out.geometry))
    print(f"[B] vert mean={nv.mean():.1f} median={np.median(nv):.0f} max={nv.max():.0f}", flush=True)

    # 面积对账
    dl = gpd.read_parquet(DLTB).to_crs(UTM)
    M = {"01": "耕地", "02": "园地", "03": "林地", "04": "草地", "05": "建筑", "06": "建筑",
         "07": "建筑", "08": "建筑", "09": "建筑", "10": "建筑", "11": "水体", "12": "荒漠"}
    dl["k"] = dl["DLBM"].astype(str).str[:2].map(M).fillna("荒漠")
    D = dl.assign(a=dl.geometry.area / 1e6).groupby("k")["a"].sum()
    Pp = out.assign(a=out.geometry.area / 1e6).groupby("label")["a"].sum()
    cls = ["耕地", "园地", "林地", "草地", "建筑", "水体", "荒漠"]
    T = pd.DataFrame({"TRUTH": D, "PRED": Pp}).reindex(cls).fillna(0)
    T["T%"] = (T["TRUTH"] / T["TRUTH"].sum() * 100).round(1)
    T["P%"] = (T["PRED"] / T["PRED"].sum() * 100).round(1)
    T["dpp"] = (T["P%"] - T["T%"]).round(1)
    print(T.round(1).to_string(), flush=True)
    print(f"[B] total PRED {T['PRED'].sum():.0f} vs TRUTH {T['TRUTH'].sum():.0f} km2 "
          f"({T['PRED'].sum()/T['TRUTH'].sum()*100-100:+.1f}%)", flush=True)
    make_previews(out_4326, t0)
    print("STAGEB_DONE", flush=True)


def vert_stats(geoms):
    nv = []
    for g in geoms:
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        tot = 0
        for p in polys:
            tot += len(p.exterior.coords) - 1
            for r in p.interiors:
                tot += len(r.coords) - 1
        nv.append(tot)
    return np.array(nv, float)


def qa_and_fill(out, cbgeo, t0):
    """无空白/无空心 QA: county.difference(union(parcels)) -> gaps; 真gap(>1m²非边界sliver)填补.
    填补法: gap多边形按底层idmap多数票定类成新parcel, 加入 out。返回 (out_filled, real_gaps, real_holes)。"""
    import geopandas as gpd
    import pandas as pd
    from shapely import unary_union, make_valid
    from shapely.geometry import Polygon
    print(f"[QA] computing gaps ({time.time()-t0:.0f}s)...", flush=True)
    cov = unary_union(out.geometry.values)
    cov = make_valid(cov)
    gaps = cbgeo.difference(cov)
    from shapely.geometry import MultiPolygon
    glist = list(gaps.geoms) if gaps.geom_type == "MultiPolygon" else ([gaps] if not gaps.is_empty else [])
    # 边界sliver: 贴县界(与县界边界距离~0)且 <1 m²
    cbb = cbgeo.boundary
    real_gaps = []; sliver_n = 0; sliver_a = 0.0
    for gp in glist:
        if gp.is_empty or gp.area <= 0:
            continue
        if gp.area < 1.0 and gp.distance(cbb) < 0.5:
            sliver_n += 1; sliver_a += gp.area; continue
        real_gaps.append(gp)
    tot_gap_a = sum(g.area for g in real_gaps)
    print(f"[QA] gaps: {len(real_gaps)} real (>1m² or non-edge), total {tot_gap_a:.1f} m² | "
          f"边界sliver {sliver_n} ({sliver_a:.2f} m²)", flush=True)
    if real_gaps:
        big = sorted(real_gaps, key=lambda x: -x.area)[:5]
        print("[QA]   largest gaps m²: " + ", ".join(f"{g.area:.1f}" for g in big), flush=True)

    # 空心洞分类(快): county.difference(cov) 已捕获所有未覆盖区域(含空心洞)。无需逐洞 difference(cov)
    # (那对巨型 union 极慢)。改为: 用 STRtree 把每个 real_gap 判为"洞"(被单个parcel包住)还是"图斑间空白"。
    # 这只用于 QA 着色; 填补对两类一视同仁(都来自 real_gaps, 已是真·未覆盖几何)。
    from shapely import STRtree
    tree = STRtree(out.geometry.values)
    real_holes = []          # 洞(橙): gap 落在某 parcel 内部(被其外环包住)
    inter_gaps = []          # 图斑间空白(红): 其余
    for gp in real_gaps:
        rp = gp.representative_point()
        cand = tree.query(rp)                       # 候选 parcel(bbox 命中)
        enclosed = False
        for ci in cand:
            g = out.geometry.values[int(ci)]
            # 洞: gap 的代表点落在某 parcel 的 exterior 内(被该 parcel 包围)
            try:
                from shapely.geometry import Polygon as _Poly
                ext = _Poly(g.exterior) if g.geom_type == "Polygon" else None
                if ext is not None and ext.contains(rp):
                    enclosed = True; break
            except Exception:
                pass
        (real_holes if enclosed else inter_gaps).append(gp)
    real_gaps = inter_gaps                            # 红只画图斑间空白
    hole_a = sum(h.area for h in real_holes)
    print(f"[QA] classified: inter-parcel gaps {len(real_gaps)} | enclosed holes {len(real_holes)} "
          f"({hole_a:.1f} m²)", flush=True)

    # 填补: real_gaps + real_holes(两类都是真·未覆盖几何)
    fill_polys = list(real_gaps) + list(real_holes)
    if fill_polys:
        labs = majority_class(fill_polys)
        add = gpd.GeoDataFrame({"label": labs}, geometry=fill_polys, crs=out.crs)
        add = add[add.geometry.geom_type.isin(["Polygon"]) & (add.geometry.area > 1.0)].reset_index(drop=True)
        n0 = len(out)
        out = gpd.GeoDataFrame(
            pd.concat([out[["label", "geometry"]], add[["label", "geometry"]]], ignore_index=True),
            geometry="geometry", crs=out.crs)
        print(f"[QA] filled {len(add)} gap/hole parcels (-> {len(out)} total, was {n0})", flush=True)
        # 复验
        cov2 = make_valid(unary_union(out.geometry.values))
        g2 = cbgeo.difference(cov2)
        gl2 = list(g2.geoms) if g2.geom_type == "MultiPolygon" else ([g2] if not g2.is_empty else [])
        rem = sum(x.area for x in gl2 if x.area >= 1.0 and x.distance(cbb) >= 0.5)
        print(f"[QA] re-check after fill: residual real-gap {rem:.1f} m² (≈0 expected)", flush=True)
    else:
        print("[QA] no real gap/hole to fill (clean coverage)", flush=True)
    return out, real_gaps, real_holes


def make_qa_fig(out, cbgeo, gaps, holes, t0):
    """全县QA图: 灰底parcel + 红(图斑间空白) + 橙(空心洞)高亮。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(14, 15))
    out.to_crs(4326).plot(ax=ax, facecolor="0.82", edgecolor="0.55", linewidth=0.03)
    n_g = len(gaps); a_g = sum(g.area for g in gaps)
    n_h = len(holes); a_h = sum(h.area for h in holes)
    if gaps:
        gpd.GeoSeries(gaps, crs=UTM).to_crs(4326).plot(ax=ax, facecolor="red", edgecolor="red", linewidth=0.8, zorder=5)
    if holes:
        gpd.GeoSeries(holes, crs=UTM).to_crs(4326).plot(ax=ax, facecolor="orange", edgecolor="orange", linewidth=0.8, zorder=6)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Yuzhong QA — gaps(red, n=%d/%.0f m²) holes(orange, n=%d/%.0f m²) BEFORE fill" %
                 (n_g, a_g, n_h, a_h), fontsize=13)
    ax.legend(handles=[Patch(facecolor="0.82", label="parcel"),
                       Patch(facecolor="red", label="gap"), Patch(facecolor="orange", label="hole")],
              loc="lower left", fontsize=11)
    p = OUTDIR + "/yuzhong_FFL_QA.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[QA] QA fig -> {p}", flush=True)


def majority_class(polys):
    """每个填补多边形, 用底层 /4 idmap 在其bbox内的多数parcel类(经cls_of)定类; 退化时用最近邻类."""
    import rasterio.features
    from affine import Affine
    from dino_parcel_export import NAME_ZH
    idmap = np.load(ART + "/idmap4.npy", mmap_mode="r")
    cls_of = {int(k): int(v) for k, v in json.load(open(ART + "/cls_of.json")).items()}
    meta = json.load(open(ART + "/meta.json"))
    a_utm = None  # polys are in UTM; idmap transform is in 3857 -> need 3857 polys
    import geopandas as gpd
    gp = gpd.GeoSeries(polys, crs=UTM).to_crs(meta["crs"])
    a = Affine(*meta["tr4"]); inv = ~a
    labs = []
    arr = np.asarray(idmap)
    Hh, Ww = arr.shape
    for geom in gp.values:
        cx, cy = geom.representative_point().coords[0]
        col, row = inv * (cx, cy)
        col = int(min(max(col, 0), Ww - 1)); row = int(min(max(row, 0), Hh - 1))
        # 3x3 邻域多数 parcel id
        r0 = max(0, row - 1); r1 = min(Hh, row + 2); c0 = max(0, col - 1); c1 = min(Ww, col + 2)
        patch = arr[r0:r1, c0:c1].ravel()
        ids = patch[patch > 0]
        if len(ids):
            vals, cnts = np.unique(ids, return_counts=True)
            pid = int(vals[cnts.argmax()])
            cid = cls_of.get(pid)
            labs.append(NAME_ZH[cid] if cid else "耕地")
        else:
            labs.append("耕地")
    return labs


def make_previews(g4, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import Patch
    import subprocess
    import geopandas as gpd
    from shapely import unary_union, make_valid
    fl = subprocess.check_output(["fc-list", ":lang=zh", "file"]).decode().split("\n")
    font = fm.FontProperties(fname=fl[0].split(":")[0].strip())
    # 全县预览
    fig, ax = plt.subplots(figsize=(14, 15))
    for c, h in HEX.items():
        s = g4[g4.label == c]
        if len(s):
            s.plot(ax=ax, facecolor=h, edgecolor="0.5", linewidth=0.04)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Yuzhong FFL FINAL — topology-aware FFL (shared-edge, 0 overlap, no seams), %d polys" % len(g4), fontsize=13)
    lg = ax.legend(handles=[Patch(facecolor=h, label=c) for c, h in HEX.items()], loc="lower left", fontsize=12)
    for t in lg.get_texts():
        t.set_fontproperties(font)
    p1 = OUTDIR + "/yuzhong_FFL_FINAL_preview.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[B] preview -> {p1}", flush=True)
    # 田块放大 (南部, 跨多个块边检查无缝)
    W = (104.18, 35.78, 104.30, 35.88)
    z = g4.cx[W[0]:W[2], W[1]:W[3]]
    fig, ax = plt.subplots(figsize=(15, 12))
    if len(z):
        z.plot(ax=ax, color=z["rgb_hex"].values, edgecolor="black", linewidth=0.35)
    ax.set_xlim(W[0], W[2]); ax.set_ylim(W[1], W[3]); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("FFL FINAL zoom — straight FFL edges, no /4 staircase, no block seams", fontsize=12)
    p2 = OUTDIR + "/yuzhong_FFL_FINAL_terrace.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[B] terrace -> {p2}", flush=True)


if __name__ == "__main__":
    main()
