"""yz_postproc — 矢量成品标准后处理: 孤立细长线条(sliver)清理.

标准末步, 所有矢量成品(SMOOTH/SMOOTH2/SMOOTH3...)都应跑一遍 eliminate_slivers.

判据(按"细长度"非纯面积, 避免误删真实小田块):
  - 平均宽度  w = area / perimeter < W_MIN  (默认 2.0m; 线状物 w 很小, 紧凑地块 w 大)
    例: 100m 长 4m 宽的田埂残片 area=400 peri≈208 -> w≈1.9m 判 sliver;
        20m×20m 真田块 area=400 peri=80 -> w=5m 不判.
  - 或 Polsby-Popper 紧凑度 PP = 4πA/P² < PP_MIN(默认0.05) 且 area < A_MAX(默认很小)
    -> 极不紧凑(蜿蜒细线)的小图斑也判 sliver.
处理 = eliminate(并入邻块)不是删:
  - 每个 sliver 并入与它"共享边界最长"的相邻地块(union, 取邻块的 class)-> 保持零重叠/无空白覆盖.
  - 无邻的(truly isolated)保留(定其底层类), 不删(否则会留空白).
  - 迭代多轮(并完可能暴露新 sliver), 直到收敛或达 max_rounds.

在米制 CRS(UTM 32648)下运行(宽度/周长才有物理意义).
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely import make_valid, get_num_coordinates
from shapely.ops import unary_union


def is_sliver(geom, w_min, pp_min, a_max):
    """True if geom 判为**线状** sliver(细 *且* 长, 不误删小方田块).

    关键: 单看平均宽度 w 会误删小方块(5×5m 田块 w=1.25m 但它是真田块, 不是线).
    线状物的本质是"细 AND 不紧凑(细长)". 用两条独立判据:
      (1) w < w_min  且  PP < PP_LINE(0.30)  —— 细 *且* 偏长条(排除小方块, 方块PP≈0.785).
      (2) PP < pp_min 且 a < a_max          —— 极不紧凑(蜿蜒细线)的小图斑.
    退化(零周长/零面积)直接判 sliver.
    """
    a = geom.area
    p = geom.length
    if p <= 0 or a <= 0:
        return True  # degenerate
    w = a / p                        # 平均宽度 (米)
    pp = 4.0 * np.pi * a / (p * p)   # Polsby-Popper 紧凑度 [0,1]; 圆=1, 方≈0.785, 细长→0
    PP_LINE = 0.30                   # 紧凑度上界: 低于此才算"长条"(方块0.785远高于此, 不误判)
    if w < w_min and pp < PP_LINE:
        return True
    if pp < pp_min and a < a_max:
        return True
    return False


def eliminate_slivers(gdf, class_col="class_id", w_min=2.0, pp_min=0.05, a_max=2000.0,
                      max_rounds=2, attr_cols=("label", "label_en", "rgb_hex"), verbose=True):
    """孤立细长线条清理(标准后处理). gdf 须为米制 CRS(UTM). 返回 (out_gdf, report).

    并入策略: sliver 并入"共享边界最长"的邻块(取邻块属性), 保持零重叠/无空白.
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
        # 把要并入的 sliver -> target 邻块 的映射; target 再吸收 sliver 几何
        absorb = {}          # target_idx -> [sliver geoms]
        consumed = set()
        merged = 0
        for si in sliv_idx:
            sg = geoms[si]
            # 候选邻块: 与 sliver 相交(touch/overlap)的非 sliver、非已消耗块
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
                n_isolated_kept += 1  # 无邻, 保留
        # 重建: target 吸收其 sliver, 被消耗的 sliver 移除
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
    """仅标记(不处理) sliver, 返回 boolean mask. 用于出清理前后对比图."""
    return np.array([is_sliver(gm, w_min, pp_min, a_max) for gm in gdf.geometry.values])
