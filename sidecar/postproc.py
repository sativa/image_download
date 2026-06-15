"""postproc — 矢量成品的标准 GIS 收尾(region-agnostic).

所有"模型→栅格 idmap→矢量化"的产品(榆中或任意新区域)矢量化后, 都应跑同一套标准收尾,
得到 **无缝(零重叠/无空白)+ 拓扑有效 + 字段标准化** 的成品。本模块就是这套标准收尾,
不绑定任何具体县/类别 schema; 县界、类别表、CRS 都是参数。

标准收尾顺序(见 `run_postproc`):
    fix_invalid -> eliminate_slivers -> fill_gaps_holes -> drop_tiny_holes -> fix_invalid -> standardize

其中 eliminate_slivers(治"悬空线段"伪影: Chaikin/平滑在拓扑节点留的极细多边形)与
drop_tiny_holes(治"微洞/微楔"伪影: 近零退化 interior ring, 出图=小点/环)是平滑成品收尾的关键两步。

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
def is_sliver(geom, w_min, sliver_max_area=100.0, min_area=None):
    """True if geom 判为 sliver(平滑/裁切在拓扑节点留的极细小多边形 = 出图悬空线段)。

    判据 = **细 且 小**: 平均宽度 w = area/perimeter < w_min  且  面积 a < sliver_max_area。
      - 细(w 小): Chaikin/平滑在节点楔出的细多边形, 渲染 boundary 时细长闭环=漂浮线段。
      - 小(a < sliver_max_area, 默认 100m²): **关键, 用面积上界保护真线性地物**——
        真·田间路/长梯田虽细(w 小)但**面积大**(贯通几十~几百米), a >= 100m² -> 不判 sliver, 不动。
        而 Chaikin 微楔面积都很小(全产品 median 7m²), 稳稳落入 a<100。
      用宽度 w(而非 Polsby-Popper)当宽度判据是为了不误删紧凑小方田块(方块 w 大)。
    `min_area`(给了时): 面积绝对过小的碎屑无条件判 sliver(无视宽度)。
    退化(零周长/零面积)直接判 sliver。

    收敛性: "细 且 面积<100m²" 的小楔并入大邻块后, 结果是大块(a 远超 100)-> 不再是 sliver,
    所以迭代很快收敛(不像纯宽度判据会反复把"细但大"的合并结果再判成 sliver)。

    其中 w = area / perimeter 平均宽度(米); 细线/微楔 w 很小, 紧凑地块 w 大。
    """
    a = geom.area
    p = geom.length
    if p <= 0 or a <= 0:
        return True  # degenerate
    if min_area is not None and a < min_area:
        return True                        # 面积绝对过小的碎屑
    w = a / p
    return (w < w_min) and (a < sliver_max_area)   # 细 且 小 = 悬空微楔


def eliminate_slivers(gdf, w_min=1.5, sliver_max_area=100.0, min_area=None,
                      isolated_drop_area=3.0, max_rounds=8, touch_tol=0.9,
                      vertex_cap=100000,
                      attr_cols=("class_id", "label", "label_en", "rgb_hex"), verbose=True):
    """细小楔(sliver)清理: **并入**相邻地块(保无缝), 孤立极小直接删, 迭代到收敛。

    gdf 须为 **米制 CRS**(UTM), 否则 w/面积无物理意义。返回 (out_gdf, report)。

    判据见 `is_sliver`: sliver = **细(w=area/peri < w_min, 默认 1.5m) 且 小(a < sliver_max_area,
    默认 100m²)**。面积上界保护真线性地物(田间路/长梯田虽细但面积大 -> 不动); 只清 Chaikin 微楔。
    "细且小"使迭代快速收敛: 小楔并入大邻块 -> 结果面积远超 100m² -> 不再是 sliver。

    邻块选择(为何不用精确"最长共享边"): Chaikin 平滑后图斑顶点极多(本数据单图斑最高 5M 顶点、
    全县 5600 万顶点), 逐对 boundary∩boundary 求最长共享边要数百秒/轮, 不可行。改用等效且极快的:
      - 候选邻块由 STRtree bbox query 给出(瞬时)。
      - 对**顶点 < vertex_cap 的候选**向量化算 sliver 代表点(point_on_surface)到候选距离;
        < touch_tol(默认 0.9m)即判**相邻**。相邻候选里取**面积最大者**为并入目标(面积最大≈
        共享边最长的稳健代理, 且把 sliver 的类别归给主邻块)。
      - 极少数超大候选(顶点 >= vertex_cap, 如 5M 顶点的背景大图斑): 精确测距不可行 -> 淘汰法,
        仅当无 treatable 相邻块时并入"最大的 giant bbox 候选"(细楔本就坐落其上)。
      - 真·孤立(无任何相邻)且 a < isolated_drop_area(默认 3m²)的极小碎屑: 直接删(本就悬空)。
      - 迭代 max_rounds(默认 8)轮, 直到收敛(merged==0 或本轮 sliver 增量 < 1%)。
    union 吸收保证零重叠/无空白覆盖(并入哪个相邻块都无缝; 面积选择只影响类别归属)。
    """
    g = gdf.copy().reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    report = {"w_min": w_min, "sliver_max_area": sliver_max_area, "min_area": min_area,
              "isolated_drop_area": isolated_drop_area, "touch_tol": touch_tol, "rounds": [],
              "n_before": int(len(g)), "area_before_km2": float(g.geometry.area.sum() / 1e6)}
    total_elim = 0
    total_elim_area = 0.0
    total_dropped = 0
    total_dropped_area = 0.0
    n_isolated_kept = 0
    prev_slivers = None

    for rnd in range(max_rounds):
        geoms = g.geometry.values
        sliv_mask = np.array([is_sliver(gm, w_min, sliver_max_area, min_area) for gm in geoms])
        sliv_idx = np.where(sliv_mask)[0]
        if len(sliv_idx) == 0:
            report["rounds"].append({"round": rnd + 1, "slivers": 0, "merged": 0, "dropped": 0})
            break
        areas = shapely.area(geoms)
        nverts = shapely.get_num_coordinates(geoms)
        reppt = shapely.point_on_surface(geoms)        # 代表点, 必在图斑内部
        tree = shapely.STRtree(geoms)
        # bbox 候选对 (瞬时): qi=sliver 局部下标, qj=候选全局下标
        qi, qj = tree.query(geoms[sliv_idx])
        si_arr = sliv_idx[qi]
        tj_arr = qj.astype(np.int64)
        keep_pair = (si_arr != tj_arr) & (~sliv_mask[tj_arr])   # 去自交 + 候选须是非-sliver
        si_arr, tj_arr = si_arr[keep_pair], tj_arr[keep_pair]
        # 相邻判据 = 代表点(在 sliver 内部)到候选多边形距离 < touch_tol。
        # 候选分两类:
        #  - treatable(顶点 < vertex_cap): 一次向量化精确测距(快)。
        #  - giant(顶点 >= vertex_cap, 如 5M 顶点的背景大图斑, 全县仅个位数): 任何精确几何 op
        #    都要数十秒~数分钟, 不可逐对/向量化测距。这类候选用**淘汰法**: 仅当某 sliver 找不到
        #    任何 treatable 相邻块时, 才把它并入"bbox 候选里最大的 giant"(它本就是坐落在该背景上的
        #    细楔; union 后 explode 兜底, 万一未真相邻则该片下一轮再处理/或保留)。
        treatable = nverts[tj_arr] < vertex_cap
        dist = np.full(len(si_arr), np.inf)
        if treatable.any():
            dist[treatable] = shapely.distance(reppt[si_arr[treatable]], geoms[tj_arr[treatable]])
        touching = dist < touch_tol

        from collections import defaultdict
        adj_by_sliver = defaultdict(list)        # si -> [(area, tj)]  已确认相邻的 treatable 候选
        giant_by_sliver = defaultdict(list)      # si -> [(area, tj)]  giant 候选(淘汰法 fallback)
        for k in range(len(si_arr)):
            s = int(si_arr[k]); t = int(tj_arr[k])
            if touching[k]:
                adj_by_sliver[s].append((float(areas[t]), t))
            elif not treatable[k]:
                giant_by_sliver[s].append((float(areas[t]), t))

        absorb = {}          # target_idx -> [sliver geoms]
        consumed = set()
        drop_idx = set()
        merged = 0
        dropped = 0
        for si in sliv_idx:
            si = int(si)
            sg = geoms[si]
            # 1) 相邻 treatable 候选里取面积最大者(面积最大≈共享边最长的稳健代理; 给地块类别)。
            cand = [(ar, t) for ar, t in adj_by_sliver.get(si, ()) if t not in consumed]
            tgt = max(cand)[1] if cand else -1
            if tgt >= 0:
                absorb.setdefault(tgt, []).append(sg)   # union 吸收 -> 无缝
                consumed.add(si)
                merged += 1
                total_elim_area += sg.area
            elif giant_by_sliver.get(si):
                # 2) 真邻只剩超大背景图斑: **不并入**(union 5M 顶点图斑既慢又会沿其边制造新 sliver, 死循环)。
                #    这种 sliver 是坐落在背景上的 <100m² 细楔 -> 直接删(留的微缺口落在背景边、极小)。
                drop_idx.add(si)
                dropped += 1
                total_dropped_area += sg.area
            elif sg.area < isolated_drop_area:
                drop_idx.add(si)               # 孤立极小碎屑 -> 删(本就悬空, 删不留缝)
                dropped += 1
                total_dropped_area += sg.area
            else:
                n_isolated_kept += 1
        new_geoms = list(geoms)
        for t, sgs in absorb.items():
            try:
                new_geoms[t] = make_valid(unary_union([geoms[t]] + sgs))
            except Exception:
                new_geoms[t] = geoms[t]
        gone = consumed | drop_idx
        keep = [i for i in range(len(g)) if i not in gone]
        g = g.iloc[keep].copy()
        g["geometry"] = [new_geoms[i] for i in keep]
        g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
        g = g.explode(index_parts=False).reset_index(drop=True)
        g = g[g.geometry.geom_type == "Polygon"].reset_index(drop=True)
        total_elim += merged
        total_dropped += dropped
        report["rounds"].append({"round": rnd + 1, "slivers": int(len(sliv_idx)),
                                 "merged": int(merged), "dropped": int(dropped)})
        if verbose:
            print(f"  [sliver] round {rnd+1}: {len(sliv_idx)} slivers -> merged {merged}, "
                  f"dropped(isolated tiny) {dropped}", flush=True)
        if merged == 0 and dropped == 0:
            break
        # 收敛: 本轮 sliver 数相比上一轮下降不足 1%(并入只在制造同等量新 sliver) -> 停, 不空转
        cur = int(len(sliv_idx))
        if prev_slivers is not None and cur >= prev_slivers * 0.99:
            if verbose:
                print(f"  [sliver] converged (slivers {prev_slivers}->{cur}, <1% drop), stop", flush=True)
            break
        prev_slivers = cur

    g = g.reset_index(drop=True)
    report["n_after"] = int(len(g))
    report["area_after_km2"] = float(g.geometry.area.sum() / 1e6)
    report["slivers_merged"] = int(total_elim)
    report["slivers_dropped_isolated"] = int(total_dropped)
    report["slivers_eliminated"] = int(total_elim + total_dropped)
    report["slivers_eliminated_area_m2"] = float(total_elim_area + total_dropped_area)
    report["slivers_dropped_area_m2"] = float(total_dropped_area)
    report["slivers_eliminated_area_pct"] = float(
        (total_elim_area + total_dropped_area) / (report["area_before_km2"] * 1e6) * 100
    ) if report["area_before_km2"] > 0 else 0.0
    report["isolated_kept"] = int(n_isolated_kept)
    return g, report


def find_slivers(gdf, w_min=1.5, sliver_max_area=100.0, min_area=None):
    """仅标记(不处理) sliver, 返回 boolean mask。用于出清理前后对比图。"""
    return np.array([is_sliver(gm, w_min, sliver_max_area, min_area)
                     for gm in gdf.geometry.values])


# ---------------------------------------------------------------------------
# 空白图斑 / 空心洞 修补
# ---------------------------------------------------------------------------
def fill_gaps_holes(gdf, boundary=None, min_gap_area=1.0, vertex_cap=100000,
                    attr_cols=("class_id", "label", "label_en", "rgb_hex"),
                    verbose=True):
    """修补"空白图斑": 图斑间空隙 + 县界内未覆盖区 + 空心洞 -> 按邻块**多数票**填补。

    用户要的"空白图斑标准修复"。两类空白都修(都保证无缝):
      (1) 县界内未被任何图斑覆盖的区域: `boundary.difference(union(gdf))`(给了 boundary)。
      (2) coverage 内部的空心洞(union 后留的 interior ring): 即使没 boundary 也能修。
    填补类别 = 与该空白**共享边界最长**的邻块类别(多数票/最长共享边), 把空白几何合并进该邻块。
    无邻的孤立空白(理论上不该有)按面积阈值 min_gap_area 丢弃(太小)或单独成块(给底层类 None)。

    gdf 须为米制 CRS。boundary: 单个 shapely 几何(已在同 CRS), None 则只修内部空心洞。
    返回 (out_gdf, report)。

    ⚠️ 性能护栏: 本步靠全量 `unary_union` 求 coverage 并集来找空隙/空心洞。当存在**超大图斑**
    (顶点 >= vertex_cap, 如 Chaikin 后 5M 顶点的背景图斑)时, 该 union 要数分钟且无 boundary 时
    收益有限(内部空心洞交给 drop_tiny_holes 清更稳)。故: **存在超大图斑且 boundary 为 None ->
    直接跳过本步**(返回原 gdf, 不做 union)。
    """
    g = gdf.copy().reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    report = {"n_before": int(len(g)), "gaps_found": 0, "gaps_filled": 0,
              "gap_area_m2": 0.0, "min_gap_area": min_gap_area, "skipped": False}

    if boundary is None and bool((shapely.get_num_coordinates(g.geometry.values) >= vertex_cap).any()):
        report["skipped"] = True
        report["n_after"] = int(len(g))
        if verbose:
            print("  [gap-hole] skip: 存在超大图斑且无 boundary -> 全量 union 不经济, "
                  "空心洞由 drop_tiny_holes 处理", flush=True)
        return g, report

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
# 微洞 (近零退化 interior ring) 清理
# ---------------------------------------------------------------------------
def drop_tiny_holes(gdf, max_drop_area=30.0, keep_if_nested=True, verbose=True):
    """删掉每个 polygon 里的**伪影微洞**(退化 interior ring), 外圈填回; 真嵌套/真空隙洞保留。

    伪影来源: Chaikin/平滑在拓扑节点处留下退化的 interior ring(全产品 8.8 万+, median≈0.0003 m²,
    亚像素看不见)。出图 = 小点/小环。删洞 = 把洞填回所属外圈, **不改外边界/不动邻块** ->
    不引入新重叠/缝(无缝守恒); 填回的是被伪环挖空的那点亚像素面积。

    判据(两条都满足才删, **守住面积守恒 + 无缝**):
      (1) **面积 < max_drop_area(默认 30m²)**: 只删微洞。大洞**一律保留** —— 因为大的非嵌套洞
          往往是**真空隙**(被背景大图斑覆盖, 但其代表点不在洞内, 所以测不到嵌套), 填回会盖到背景
          大图斑 -> 制造重叠。实测榆中: 非嵌套洞里 87816 个 <1m²(共 30m², 真伪影), 其余 ~600 个
          大洞共 8.25 km²(真空隙, 必须留)。用 30m² 上界把这两类干净分开。
      (2) keep_if_nested=True 时: 洞内若落有别的 parcel 代表点 -> 真嵌套地块, **保留**(填回会盖住它)。
          微洞(<30m²)基本套不下别的地块, 这条几乎不触发, 但留作双保险。

    gdf 须为 **米制 CRS**(UTM)。返回 (out_gdf, report)。
    """
    from shapely.geometry import Polygon
    g = gdf.copy().reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    geoms = g.geometry.values
    report = {"max_drop_area": max_drop_area, "n_poly": int(len(g)), "holes_before": 0,
              "holes_dropped": 0, "holes_kept_nested": 0, "holes_kept_large": 0,
              "hole_area_dropped_m2": 0.0}

    cent = shapely.point_on_surface(geoms)     # 代表点索引: 判洞内有没有别的 parcel(嵌套)
    ctree = shapely.STRtree(cent)

    new_geoms = list(geoms)
    n_drop = 0
    n_keep_nested = 0
    n_keep_large = 0
    area_drop = 0.0
    n_before = 0
    for i, gm in enumerate(geoms):
        if not gm.interiors:
            continue
        keep_rings = []
        changed = False
        for ring in gm.interiors:
            n_before += 1
            hole = Polygon(ring)
            ha = hole.area if hole.is_valid else 0.0
            if ha >= max_drop_area:            # 大洞: 真空隙/真嵌套 -> 保留(填回会制造重叠)
                keep_rings.append(ring); n_keep_large += 1; continue
            nested = False                     # 微洞内有没有别的 parcel 代表点? 有 -> 真嵌套, 保留
            if keep_if_nested and ha > 0:
                for j in ctree.query(hole):
                    j = int(j)
                    if j != i and hole.contains(cent[j]):
                        nested = True
                        break
            if nested:
                keep_rings.append(ring); n_keep_nested += 1; continue
            changed = True                     # 微洞伪影 -> 删(填回外圈)
            n_drop += 1
            area_drop += ha
        if changed:
            new_geoms[i] = Polygon(gm.exterior, keep_rings)
    g["geometry"] = new_geoms
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    report["holes_before"] = int(n_before)
    report["holes_dropped"] = int(n_drop)
    report["holes_kept_nested"] = int(n_keep_nested)
    report["holes_kept_large"] = int(n_keep_large)
    report["hole_area_dropped_m2"] = float(area_drop)
    if verbose:
        print(f"  [hole] {n_before} interior rings -> dropped {n_drop} micro-artifact holes "
              f"(<{max_drop_area} m², {area_drop:.1f} m² filled back), "
              f"kept {n_keep_nested} nested + {n_keep_large} large(real gaps)", flush=True)
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
    # UTM->4326 重投影的浮点舍入可能给极少数图斑引入自交 -> 最后再 make_valid 一遍, 保证出文件 all-valid。
    inv = ~g.geometry.is_valid
    if bool(inv.any()):
        g.loc[inv, "geometry"] = g.loc[inv, "geometry"].apply(make_valid)
        g = g.explode(index_parts=False).reset_index(drop=True)
        g = g[g.geometry.geom_type == "Polygon"].reset_index(drop=True)
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
                 sliver_kw=None, gap_kw=None, hole_kw=None, skip_gaps=False, verbose=True):
    """**所有模型矢量成品的标准收尾流程**(region-agnostic), 串起标准末步:

        fix_invalid -> eliminate_slivers -> fill_gaps_holes -> drop_tiny_holes -> fix_invalid -> standardize

    ⚠️ skip_gaps=True: **跳过 fill_gaps_holes**。当输入已是**拓扑无缝 coverage**(如 topojson 重建的
    全县 coverage, 零真空隙)且含**大量合法内部洞**(被路网圈住的田 = 巨型背景图斑的上千 interior ring)时,
    fill_gaps_holes 会把这些"洞"误当空白填进邻块, 造成**面积重复计**(实测榆中草地巨型连通域 +90km² 虚高,
    所有 1494 个"gap"其实是已被其它 parcel 覆盖的内部洞)。这种输入应 skip_gaps=True(coverage 本就无缝,
    无需填; drop_tiny_holes 仍清退化微洞)。仅当输入可能有**真空隙**(per-cell 拼接留白缝等)时才开 fill。

    设计意图: 任何 backend(cropland 二分类 / parcel watershed / SAM3 / landcover)出的
    原始矢量 idmap, 跑完这一套都得到同口径的无缝标准成品(零重叠/无空白/拓扑有效/字段统一)。

    末步两个去伪影步(针对 Chaikin/平滑节点微楔, 默认参数即够狠, 治用户图里的"悬空线段"):
      - eliminate_slivers: sliver = 细(w=area/peri<1.5m) 且 小(area<100m²); 并入面积最大相邻块
        (距离判相邻, 超大背景图斑不可并 -> 直接删 <100m² 细楔); 迭代到收敛。面积上界护住真路/长梯田。
      - drop_tiny_holes: 删 area<30m² 的退化 interior ring(微洞伪影), 外圈填回; 大洞(真空隙/真嵌套)
        保留 -> 不改外边界/不动邻块, 面积守恒 + 无缝。
    放在 fill_gaps_holes 之后(那步若无 boundary 也会把大空心洞并给邻块, 这里再清残留的微洞),
    fix_invalid/standardize 之前。

    参数:
      gdf      : 原始矢量(任意 CRS, 须带 class_id 列)。
      classes  : 类别 schema(dict 或 list, 见 standardize)。
      boundary : 县界几何(shapely, 任意输入 CRS 由调用方负责 -> 这里假定已是 utm); None 跳过空白填补的边界差集。
      utm      : 米制 CRS, 几何运算(sliver/gap/hole/面积)都在此 CRS 下做。
                 注意: 不同区域 UTM 带不同(榆中 32648 / 神池 32649), 调用方须给对带。
    返回 (out_gdf, report)。out_gdf 为 out_crs, 字段标准化。
    """
    sliver_kw = sliver_kw or {}
    gap_kw = gap_kw or {}
    hole_kw = hole_kw or {}
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
    if skip_gaps:
        report["fill_gaps_holes"] = {"skipped": True,
                                     "reason": "seamless coverage input (topojson) — interior holes are legit nested parcels, not gaps"}
        if verbose:
            print("  [gap-hole] SKIPPED (skip_gaps=True): 输入为无缝 coverage, 内部洞=合法嵌套田块, 不填", flush=True)
    else:
        gu, r_gap = fill_gaps_holes(gu, boundary=bnd_utm, verbose=verbose, **gap_kw)
        report["fill_gaps_holes"] = r_gap

    gu, r_hole = drop_tiny_holes(gu, verbose=verbose, **hole_kw)
    report["drop_tiny_holes"] = r_hole

    gu = fix_invalid(gu)
    report["steps"].append({"fix_invalid_2": int(len(gu))})

    out = standardize(gu, classes, out_crs=out_crs, utm=utm)
    report["n_final"] = int(len(out))
    report["area_final_km2"] = float(out.to_crs(utm).geometry.area.sum() / 1e6)
    if verbose:
        print(f"[postproc] DONE: {report['n_final']} polys, "
              f"{report['area_final_km2']:.1f} km2 -> {out_crs}", flush=True)
    return out, report
