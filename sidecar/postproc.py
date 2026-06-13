"""postproc — 矢量成品的标准 GIS 收尾(region-agnostic).

所有"模型→栅格 idmap→矢量化"的产品(榆中或任意新区域)矢量化后, 都应跑同一套标准收尾,
得到 **无缝(零重叠/无空白)+ 拓扑有效 + 字段标准化** 的成品。本模块就是这套标准收尾,
不绑定任何具体县/类别 schema; 县界、类别表、CRS 都是参数。

标准收尾顺序(见 `run_postproc`):
    fix_invalid -> eliminate_slivers -> fill_gaps_holes -> fix_invalid -> standardize

每个函数都在 **米制 CRS(默认 UTM)** 下做几何运算(宽度/周长/面积才有物理意义),
最后 standardize 统一转回 EPSG:4326 输出。

判据/阈值含义见各函数 docstring。
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely import make_valid
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# 几何有效性
# ---------------------------------------------------------------------------
def fix_invalid(gdf, geom_col="geometry"):
    """make_valid 修自相交/退化几何, 丢空/None, explode 到单 Polygon。

    判据/做法:
      - 任何 `is_valid==False` 的几何(自相交、bowtie、重复点)走 shapely.make_valid。
      - make_valid 可能把一个坏 Polygon 拆成 GeometryCollection/MultiPolygon ->
        explode 成单 Polygon, 只保留面状(丢线/点碎屑)。
    这是平滑/裁切等几何操作后必跑的"卫生"步, 保证后续运算不被坏几何炸掉。
    """
    g = gdf.copy().reset_index(drop=True)
    g[geom_col] = g[geom_col].apply(lambda x: x if (x is not None and x.is_valid) else make_valid(x)
                                    if x is not None else None)
    g = g[g[geom_col].notna() & ~g[geom_col].is_empty].reset_index(drop=True)
    g = g.explode(index_parts=False).reset_index(drop=True)
    g = g[g.geometry.geom_type == "Polygon"].reset_index(drop=True)
    return g


# ---------------------------------------------------------------------------
# sliver(细长线状碎屑)清理
# ---------------------------------------------------------------------------
def is_sliver(geom, w_min, pp_min, a_max):
    """True if geom 判为**线状** sliver(细 *且* 长, 不误删真实小方田块)。

    关键: 单看平均宽度 w 会误删小方块(5x5m 田块 w=1.25m 但它是真田块, 不是线)。
    线状物的本质是"细 AND 不紧凑(细长)"。用两条独立判据:
      (1) w < w_min  且  PP < PP_LINE(0.30)  —— 细 *且* 偏长条(排除小方块, 方块 PP≈0.785)。
      (2) PP < pp_min 且 a < a_max          —— 极不紧凑(蜿蜒细线)的小图斑。
    退化(零周长/零面积)直接判 sliver。

    其中:
      w  = area / perimeter         平均宽度 (米); 线状物 w 很小, 紧凑地块 w 大。
      PP = 4*pi*A / P^2             Polsby-Popper 紧凑度 [0,1]; 圆=1, 方≈0.785, 细长→0。
    """
    a = geom.area
    p = geom.length
    if p <= 0 or a <= 0:
        return True  # degenerate
    w = a / p
    pp = 4.0 * np.pi * a / (p * p)
    PP_LINE = 0.30
    if w < w_min and pp < PP_LINE:
        return True
    if pp < pp_min and a < a_max:
        return True
    return False


def eliminate_slivers(gdf, w_min=2.0, pp_min=0.05, a_max=2000.0, max_rounds=2,
                      attr_cols=("class_id", "label", "label_en", "rgb_hex"), verbose=True):
    """细长线状物(sliver)清理: **并入**最长共享边邻块(非删, 保无缝)。

    gdf 须为 **米制 CRS**(UTM), 否则 w/PP/a_max 无物理意义。返回 (out_gdf, report)。

    判据见 `is_sliver`(w<w_min 且偏长条, 或极不紧凑的小图斑)。
    并入策略(保证零重叠/无空白覆盖):
      - 每个 sliver 并入与它"共享边界最长"的相邻非-sliver 地块(取邻块属性) -> 邻块 union 吸收。
      - 无邻的(truly isolated)保留, 不删(否则会留空白)。
      - 迭代 max_rounds 轮(并完可能暴露新 sliver), 直到收敛。
    """
    g = gdf.copy().reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    report = {"w_min": w_min, "pp_min": pp_min, "a_max": a_max, "rounds": [],
              "n_before": int(len(g)), "area_before_km2": float(g.geometry.area.sum() / 1e6)}
    total_elim = 0
    total_elim_area = 0.0
    n_isolated_kept = 0

    for rnd in range(max_rounds):
        geoms = g.geometry.values
        sliv_mask = np.array([is_sliver(gm, w_min, pp_min, a_max) for gm in geoms])
        sliv_idx = np.where(sliv_mask)[0]
        if len(sliv_idx) == 0:
            report["rounds"].append({"round": rnd + 1, "slivers": 0, "merged": 0})
            break
        tree = shapely.STRtree(geoms)
        absorb = {}          # target_idx -> [sliver geoms]
        consumed = set()
        merged = 0
        for si in sliv_idx:
            sg = geoms[si]
            cand = tree.query(sg, predicate="intersects")
            best_t, best_len = -1, -1.0
            for t in cand:
                t = int(t)
                if t == si or t in consumed or sliv_mask[t]:
                    continue
                try:
                    shared = sg.boundary.intersection(geoms[t].boundary).length
                except Exception:
                    shared = 0.0
                if shared > best_len:
                    best_len, best_t = shared, t
            if best_t >= 0 and best_len > 0:
                absorb.setdefault(best_t, []).append(sg)
                consumed.add(si)
                merged += 1
                total_elim_area += sg.area
            else:
                n_isolated_kept += 1
        new_geoms = list(geoms)
        for t, sgs in absorb.items():
            try:
                new_geoms[t] = make_valid(unary_union([geoms[t]] + sgs))
            except Exception:
                new_geoms[t] = geoms[t]
        keep = [i for i in range(len(g)) if i not in consumed]
        g = g.iloc[keep].copy()
        g["geometry"] = [new_geoms[i] for i in keep]
        g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
        g = g.explode(index_parts=False).reset_index(drop=True)
        g = g[g.geometry.geom_type == "Polygon"].reset_index(drop=True)
        total_elim += merged
        report["rounds"].append({"round": rnd + 1, "slivers": int(len(sliv_idx)), "merged": int(merged)})
        if verbose:
            print(f"  [sliver] round {rnd+1}: {len(sliv_idx)} slivers -> merged {merged}", flush=True)
        if merged == 0:
            break

    g = g.reset_index(drop=True)
    report["n_after"] = int(len(g))
    report["area_after_km2"] = float(g.geometry.area.sum() / 1e6)
    report["slivers_eliminated"] = int(total_elim)
    report["slivers_eliminated_area_m2"] = float(total_elim_area)
    report["slivers_eliminated_area_pct"] = float(
        total_elim_area / (report["area_before_km2"] * 1e6) * 100) if report["area_before_km2"] > 0 else 0.0
    report["isolated_kept"] = int(n_isolated_kept)
    return g, report


def find_slivers(gdf, w_min=2.0, pp_min=0.05, a_max=2000.0):
    """仅标记(不处理) sliver, 返回 boolean mask。用于出清理前后对比图。"""
    return np.array([is_sliver(gm, w_min, pp_min, a_max) for gm in gdf.geometry.values])


# ---------------------------------------------------------------------------
# 空白图斑 / 空心洞 修补
# ---------------------------------------------------------------------------
def fill_gaps_holes(gdf, boundary=None, min_gap_area=1.0, attr_cols=("class_id", "label", "label_en", "rgb_hex"),
                    verbose=True):
    """修补"空白图斑": 图斑间空隙 + 县界内未覆盖区 + 空心洞 -> 按邻块**多数票**填补。

    用户要的"空白图斑标准修复"。两类空白都修(都保证无缝):
      (1) 县界内未被任何图斑覆盖的区域: `boundary.difference(union(gdf))`(给了 boundary)。
      (2) coverage 内部的空心洞(union 后留的 interior ring): 即使没 boundary 也能修。
    填补类别 = 与该空白**共享边界最长**的邻块类别(多数票/最长共享边), 把空白几何合并进该邻块。
    无邻的孤立空白(理论上不该有)按面积阈值 min_gap_area 丢弃(太小)或单独成块(给底层类 None)。

    gdf 须为米制 CRS。boundary: 单个 shapely 几何(已在同 CRS), None 则只修内部空心洞。
    返回 (out_gdf, report)。
    """
    g = gdf.copy().reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    report = {"n_before": int(len(g)), "gaps_found": 0, "gaps_filled": 0,
              "gap_area_m2": 0.0, "min_gap_area": min_gap_area}

    union = unary_union(g.geometry.values)
    gaps = []
    # (1) boundary-difference gaps (县界内未覆盖)
    if boundary is not None:
        bnd = make_valid(boundary)
        diff = bnd.difference(union)
        if not diff.is_empty:
            gaps += list(getattr(diff, "geoms", [diff]))
    # (2) interior holes of the coverage union (空心洞)
    poly_union = list(getattr(union, "geoms", [union]))
    from shapely.geometry import Polygon
    for poly in poly_union:
        if getattr(poly, "geom_type", "") != "Polygon":
            continue
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.is_valid and not hole.is_empty:
                gaps.append(hole)
    gaps = [gg for gg in gaps if getattr(gg, "geom_type", "") == "Polygon"
            and gg.area >= min_gap_area]
    report["gaps_found"] = len(gaps)
    if not gaps:
        report["n_after"] = int(len(g))
        return g, report

    geoms = g.geometry.values
    tree = shapely.STRtree(geoms)
    absorb = {}      # target_idx -> [gap geoms]
    n_filled = 0
    gap_area = 0.0
    for gg in gaps:
        cand = tree.query(gg, predicate="intersects")
        best_t, best_len = -1, -1.0
        for t in cand:
            t = int(t)
            try:
                shared = gg.boundary.intersection(geoms[t].boundary).length
            except Exception:
                shared = 0.0
            if shared > best_len:
                best_len, best_t = shared, t
        if best_t >= 0:
            absorb.setdefault(best_t, []).append(gg)
            n_filled += 1
            gap_area += gg.area
    new_geoms = list(geoms)
    for t, ggs in absorb.items():
        try:
            new_geoms[t] = make_valid(unary_union([geoms[t]] + ggs))
        except Exception:
            new_geoms[t] = geoms[t]
    g["geometry"] = new_geoms
    g = fix_invalid(g)
    report["gaps_filled"] = n_filled
    report["gap_area_m2"] = float(gap_area)
    report["n_after"] = int(len(g))
    if verbose:
        print(f"  [gap-hole] found {len(gaps)} gaps/holes -> filled {n_filled} "
              f"({gap_area/1e4:.2f} ha) into nearest neighbour", flush=True)
    return g, report


# ---------------------------------------------------------------------------
# 字段标准化
# ---------------------------------------------------------------------------
def standardize(gdf, classes, class_col="class_id", out_crs="EPSG:4326", utm="EPSG:32648"):
    """加 gid / area_m2(UTM 米制) / label / label_en / rgb_hex, 统一 EPSG:4326 输出。

    classes: dict[class_id] -> {"label_zh","label_en","rgb"}  或
             list[(id, label_zh, label_en, (r,g,b))]  (与 dino_parcel_export.CLASSES 同构)。
    area_m2 在 `utm`(米制)下算后再转 out_crs(EPSG:4326 是度, 不能直接量面积)。
    返回标准化 GeoDataFrame(列: gid, class_id, label, label_en, rgb_hex, area_m2, geometry)。
    """
    cmap = _norm_classes(classes)
    g = gdf.copy().reset_index(drop=True)
    if g.crs is None:
        raise ValueError("standardize: gdf 必须带 CRS")
    gu = g.to_crs(utm)
    g["area_m2"] = gu.geometry.area.round(1).values
    cids = g[class_col].astype(int)
    g["label"] = [cmap.get(c, {}).get("label_zh", str(c)) for c in cids]
    g["label_en"] = [cmap.get(c, {}).get("label_en", str(c)) for c in cids]
    g["rgb_hex"] = [cmap.get(c, {}).get("rgb_hex", "#999999") for c in cids]
    g = g.to_crs(out_crs)
    g = g[[c for c in ["class_id", "label", "label_en", "rgb_hex", "area_m2", "geometry"]
           if c in g.columns or c == class_col]].copy()
    if "gid" in g.columns:
        g = g.drop(columns=["gid"])
    g.insert(0, "gid", range(1, len(g) + 1))
    return g


def _norm_classes(classes):
    """把 classes(dict 或 list-of-tuple)统一成 {id: {label_zh,label_en,rgb_hex}}。"""
    out = {}
    if isinstance(classes, dict):
        for k, v in classes.items():
            cid = int(k)
            if isinstance(v, dict):
                rgb = v.get("rgb")
                out[cid] = {"label_zh": v.get("label_zh", v.get("label", str(cid))),
                            "label_en": v.get("label_en", str(cid)),
                            "rgb_hex": v.get("rgb_hex") or (_hex(rgb) if rgb else "#999999")}
            elif isinstance(v, (list, tuple)) and len(v) >= 3:
                out[cid] = {"label_zh": v[0], "label_en": v[1], "rgb_hex": _hex(v[2])}
    else:  # list of (id, zh, en, (r,g,b))
        for row in classes:
            out[int(row[0])] = {"label_zh": row[1], "label_en": row[2], "rgb_hex": _hex(row[3])}
    return out


def _hex(rgb):
    if isinstance(rgb, str):
        return rgb
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ---------------------------------------------------------------------------
# 标准收尾流程
# ---------------------------------------------------------------------------
def run_postproc(gdf, classes, boundary=None, utm="EPSG:32648", out_crs="EPSG:4326",
                 sliver_kw=None, gap_kw=None, verbose=True):
    """**所有模型矢量成品的标准收尾流程**(region-agnostic), 串起标准末步:

        fix_invalid -> eliminate_slivers -> fill_gaps_holes -> fix_invalid -> standardize

    设计意图: 任何 backend(cropland 二分类 / parcel watershed / SAM3 / landcover)出的
    原始矢量 idmap, 跑完这一套都得到同口径的无缝标准成品(零重叠/无空白/拓扑有效/字段统一)。

    参数:
      gdf      : 原始矢量(任意 CRS, 须带 class_id 列)。
      classes  : 类别 schema(dict 或 list, 见 standardize)。
      boundary : 县界几何(shapely, 任意输入 CRS 由调用方负责 -> 这里假定已是 utm); None 跳过空白填补的边界差集。
      utm      : 米制 CRS, 几何运算(sliver/gap/面积)都在此 CRS 下做。
    返回 (out_gdf, report)。out_gdf 为 out_crs, 字段标准化。
    """
    sliver_kw = sliver_kw or {}
    gap_kw = gap_kw or {}
    report = {"steps": []}
    g = gdf
    # 统一进入米制 CRS 做几何运算
    if g.crs is None:
        raise ValueError("run_postproc: gdf 必须带 CRS")
    gu = g.to_crs(utm)

    gu = fix_invalid(gu)
    report["steps"].append({"fix_invalid_1": int(len(gu))})

    gu, r_sliv = eliminate_slivers(gu, verbose=verbose, **sliver_kw)
    report["eliminate_slivers"] = r_sliv

    bnd_utm = None
    if boundary is not None:
        bnd_utm = make_valid(boundary)
    gu, r_gap = fill_gaps_holes(gu, boundary=bnd_utm, verbose=verbose, **gap_kw)
    report["fill_gaps_holes"] = r_gap

    gu = fix_invalid(gu)
    report["steps"].append({"fix_invalid_2": int(len(gu))})

    out = standardize(gu, classes, out_crs=out_crs, utm=utm)
    report["n_final"] = int(len(out))
    report["area_final_km2"] = float(out.to_crs(utm).geometry.area.sum() / 1e6)
    if verbose:
        print(f"[postproc] DONE: {report['n_final']} polys, "
              f"{report['area_final_km2']:.1f} km2 -> {out_crs}", flush=True)
    return out, report
