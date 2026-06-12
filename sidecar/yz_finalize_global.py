"""GLOBAL 收尾(真几何裁剪版): 全局 watershed 无块边 -> 不缝合; 用真·几何 intersection 裁到县界
(替代 seam_finalize 的质心法, 因全局大地块跨界会被质心法整块丢 -> 面积亏损). 平滑 + 面积对账 + 出图。"""
import sys, time
import numpy as np
import geopandas as gpd
import pandas as pd
from collections import defaultdict
sys.path.insert(0, "/home/ps/landform/sidecar")
from dino_parcel_export import smooth_geom
from shapely import make_valid
from shapely.validation import make_valid as _mv

IN = "/mnt/sda/zf/landform/results/yuzhong_global_region.parquet"
OUT = "/mnt/sda/zf/landform/results/yuzhong_GLOBAL.parquet"
CB = "/tmp/yz_county_boundary.parquet"
DLTB = "/home/ps/landform/data/v11_dltb/620123.parquet"
UTM = "EPSG:32648"
TAG = "GLOBAL"
HEX = {"耕地": "#3cb44b", "园地": "#aaff5a", "林地": "#006400", "草地": "#bedc64",
       "水体": "#0082c8", "建筑": "#e6194b", "荒漠": "#aa8c64"}
EN = {"耕地": "cropland", "园地": "orchard", "林地": "forest", "草地": "grassland",
      "水体": "water", "建筑": "built", "荒漠": "baresoil"}
M = {"01": "耕地", "02": "园地", "03": "林地", "04": "草地", "05": "建筑", "06": "建筑",
     "07": "建筑", "08": "建筑", "09": "建筑", "10": "建筑", "11": "水体", "12": "荒漠"}


def main():
    t0 = time.time()
    g = gpd.read_parquet(IN).to_crs(UTM).reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    print(f"[gfinal] {len(g)} parcels loaded ({time.time()-t0:.0f}s)", flush=True)

    cb = gpd.read_parquet(CB).to_crs(UTM)
    cbgeo = make_valid(cb.geometry.values[0])
    print(f"[gfinal] county boundary area {cbgeo.area/1e6:.1f} km2", flush=True)

    # bbox 预筛 (只留与县界 bbox 重叠的)
    minx, miny, maxx, maxy = cbgeo.bounds
    b = g.geometry.bounds.values
    keep = ~((b[:, 2] < minx) | (b[:, 0] > maxx) | (b[:, 3] < miny) | (b[:, 1] > maxy))
    g = g[keep].reset_index(drop=True)
    print(f"[gfinal] bbox-overlap {len(g)} parcels", flush=True)

    # 真·几何裁剪: 完全在县内的(within)直接保留, 跨界的才做 intersection (省时)
    sidx_cov = g.geometry.covered_by(cbgeo)         # 完全在内
    inside = g[sidx_cov].copy()
    edgers = g[~sidx_cov].copy()
    print(f"[gfinal] fully-inside {len(inside)} | boundary {len(edgers)} -> clip ({time.time()-t0:.0f}s)", flush=True)
    if len(edgers):
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
    else:
        ec = gpd.GeoDataFrame({"label": [], "geometry": []}, crs=UTM)
    out = gpd.GeoDataFrame(pd.concat([inside[["label", "geometry"]], ec[["label", "geometry"]]], ignore_index=True), crs=UTM)
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    print(f"[gfinal] clipped -> {len(out)} polys ({time.time()-t0:.0f}s)", flush=True)

    # 平滑
    out["geometry"] = out.geometry.apply(smooth_geom)
    out["geometry"] = out.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    out = out[~out.geometry.is_empty & out.geometry.notna()].reset_index(drop=True)
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    print(f"[gfinal] smoothed {len(out)} ({time.time()-t0:.0f}s)", flush=True)

    out["label_en"] = out["label"].map(EN); out["rgb_hex"] = out["label"].map(HEX)
    out["area_m2"] = out.geometry.area.round(1); out.insert(0, "gid", range(1, len(out) + 1))
    out.to_crs("EPSG:4326").to_parquet(OUT)
    print(f"[gfinal] FINAL {len(out)} polys -> {OUT} ({time.time()-t0:.0f}s)", flush=True)

    # 面积对账
    dl = gpd.read_parquet(DLTB).to_crs(UTM)
    dl["k"] = dl["DLBM"].astype(str).str[:2].map(M).fillna("荒漠")
    D = dl.assign(a=dl.geometry.area / 1e6).groupby("k")["a"].sum()
    P = out.assign(a=out.geometry.area / 1e6).groupby("label")["a"].sum()
    cls = ["耕地", "园地", "林地", "草地", "建筑", "水体", "荒漠"]
    T = pd.DataFrame({"TRUTH": D, "PRED": P}).reindex(cls).fillna(0)
    T["T%"] = (T["TRUTH"] / T["TRUTH"].sum() * 100).round(1)
    T["P%"] = (T["PRED"] / T["PRED"].sum() * 100).round(1)
    T["dpp"] = (T["P%"] - T["T%"]).round(1)
    print(T.round(1).to_string(), flush=True)
    print(f"[gfinal] total PRED {T['PRED'].sum():.0f} vs TRUTH {T['TRUTH'].sum():.0f} km2 ({T['PRED'].sum()/T['TRUTH'].sum()*100-100:+.1f}%)", flush=True)

    # 出图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import Patch
    import subprocess
    fl = subprocess.check_output(["fc-list", ":lang=zh", "file"]).decode().split("\n")
    font = fm.FontProperties(fname=fl[0].split(":")[0].strip())
    g4 = out.to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(14, 15))
    for c, h in HEX.items():
        s = g4[g4.label == c]
        if len(s):
            s.plot(ax=ax, facecolor=h, edgecolor="0.5", linewidth=0.04)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Yuzhong GLOBAL — single-image unified watershed (/4, ridge, enhance6+Hann), %d polys" % len(g4), fontsize=13)
    lg = ax.legend(handles=[Patch(facecolor=h, label=c) for c, h in HEX.items()], loc="lower left", fontsize=12)
    for t in lg.get_texts():
        t.set_fontproperties(font)
    plt.savefig(f"/mnt/sda/zf/landform/results/yuzhong_{TAG}_preview.png", dpi=120, bbox_inches="tight", facecolor="white")
    print(f"[gfinal] preview -> /mnt/sda/zf/landform/results/yuzhong_{TAG}_preview.png", flush=True)
    # 放大图 (南部丘陵带, 看无缝/无块边)
    W = (104.18, 35.78, 104.30, 35.88)
    z = g4.cx[W[0]:W[2], W[1]:W[3]]
    fig, ax = plt.subplots(figsize=(15, 12))
    if len(z):
        z.plot(ax=ax, color=z["rgb_hex"].values, edgecolor="black", linewidth=0.3)
    ax.set_xlim(W[0], W[2]); ax.set_ylim(W[1], W[3]); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("GLOBAL zoom — seamless? (no block/piece lines)", fontsize=12)
    plt.savefig(f"/mnt/sda/zf/landform/results/yuzhong_{TAG}_terrace.png", dpi=140, bbox_inches="tight", facecolor="white")
    print("SAVED_OK", flush=True)


if __name__ == "__main__":
    main()
