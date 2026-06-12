"""榆中县级产物收尾:① 分幅缝合(只合并"跨块边界、相邻、同类"的图斑,块内田块不动)
② 对合并后的边界整体平滑(Chaikin,放在 dissolve 之后,不破坏相邻贴合)③ 裁到真实县界(620123)
④ 出预览/梯田放大图 + 对真值面积对账。输入 = yz_blocks 逐块 shapes 产物(精确贴合, 带 block 列)。"""
import sys, time, json
import numpy as np
import geopandas as gpd
import pandas as pd
from collections import defaultdict
from shapely.ops import unary_union
sys.path.insert(0, "/home/ps/landform/sidecar")
from dino_parcel_export import smooth_geom

IN = "/mnt/sda/zf/landform/results/yuzhong_enh_region.parquet"
OUT = "/mnt/sda/zf/landform/results/yuzhong_FINAL.parquet"
DLTB = "/home/ps/landform/data/v11_dltb/620123.parquet"   # 榆中三调真值
UTM = "EPSG:32648"
HEX = {"耕地": "#3cb44b", "园地": "#aaff5a", "林地": "#006400", "草地": "#bedc64",
       "水体": "#0082c8", "建筑": "#e6194b", "荒漠": "#aa8c64"}
EN = {"耕地": "cropland", "园地": "orchard", "林地": "forest", "草地": "grassland",
      "水体": "water", "建筑": "built", "荒漠": "baresoil"}
M = {"01": "耕地", "02": "园地", "03": "林地", "04": "草地", "05": "建筑", "06": "建筑",
     "07": "建筑", "08": "建筑", "09": "建筑", "10": "建筑", "11": "水体", "12": "荒漠"}
TOL = 3.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tag", default="FINAL")              # 预览图后缀
    ap.add_argument("--no-dissolve", action="store_true")  # centroid质心输出已无缝, 跳过分幅缝合
    A = ap.parse_args()
    globals()["IN"] = A.inp; globals()["OUT"] = A.out; globals()["TAG"] = A.tag
    t0 = time.time()
    g = gpd.read_parquet(A.inp).to_crs(UTM).reset_index(drop=True)
    g["geometry"] = g.geometry.buffer(0)
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    N = len(g); lab = g["label"].values
    print(f"[finalize] {N} parcels loaded ({time.time()-t0:.0f}s)", flush=True)

    if A.no_dissolve:                                             # centroid质心输出已无缝 -> 每地块独立, 不缝合(避免误并相邻不同田块)
        out = gpd.GeoDataFrame({"label": lab}, geometry=g.geometry.values, crs=UTM)
        out = out.explode(index_parts=False).reset_index(drop=True)
        out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
        print(f"[finalize] 跳过缝合(centroid已无缝): {len(out)} parcels ({time.time()-t0:.0f}s)", flush=True)
    else:
        b = g.geometry.bounds.values; blk = g["block"].values
        bb = {}                                                   # 分幅线 = 各块外框边
        for k in np.unique(blk):
            idx = np.where(blk == k)[0]; bx = b[idx]
            bb[k] = (bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max())
        xl = np.array(sorted({round(v[0], 1) for v in bb.values()} | {round(v[2], 1) for v in bb.values()}))
        yl = np.array(sorted({round(v[1], 1) for v in bb.values()} | {round(v[3], 1) for v in bb.values()}))
        def near(lo, hi, lines):
            return np.min(np.abs(lines - lo)) < TOL or np.min(np.abs(lines - hi)) < TOL
        cand = [i for i in range(N) if near(b[i, 0], b[i, 2], xl) or near(b[i, 1], b[i, 3], yl)]
        print(f"[finalize] {len(cand)} boundary-candidate parcels", flush=True)
        cg = g.iloc[cand].copy(); cg["geometry"] = cg.geometry.buffer(TOL / 2); cg["ci"] = cand
        j = gpd.sjoin(cg[["geometry", "ci"]], cg[["geometry", "ci"]], predicate="intersects", how="inner")
        par = np.arange(N)
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        m = 0
        for a, c in zip(j["ci_left"].values, j["ci_right"].values):
            if a >= c:
                continue
            if blk[a] != blk[c] and lab[a] == lab[c]:
                ra, rc = find(a), find(c)
                if ra != rc:
                    par[ra] = rc; m += 1
        root = np.array([find(i) for i in range(N)]); grp = defaultdict(list)
        for i in range(N):
            grp[root[i]].append(i)
        gv = g.geometry.values; rows = []
        for r, idxs in grp.items():
            i0 = idxs[0]
            gm = gv[i0] if len(idxs) == 1 else unary_union([gv[k] for k in idxs])
            rows.append((gm, lab[i0]))
        out = gpd.GeoDataFrame({"label": [x[1] for x in rows]}, geometry=[x[0] for x in rows], crs=UTM)
        out = out.explode(index_parts=False).reset_index(drop=True)
        out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
        print(f"[finalize] 分幅缝合 {N} -> {len(out)} (merges={m}, {time.time()-t0:.0f}s)", flush=True)

    # 平滑(合并之后, 对最终边界整体做; smooth_geom 跳过 >120 点大环避免抖动)
    out["geometry"] = out.geometry.apply(smooth_geom)
    out["geometry"] = out.geometry.buffer(0)
    out = out[~out.geometry.is_empty & out.geometry.notna()].reset_index(drop=True)
    print(f"[finalize] 平滑完成 ({time.time()-t0:.0f}s)", flush=True)

    # 裁县界(快法, 无慢 union):质心落在任一 DLTB 图斑内则在县内 → 保留; 界外丢
    # (95868 图斑的精确 union 太慢/会挂; point-in-poly 对 DLTB 图斑 sindex 剪枝, 秒级)
    dl = gpd.read_parquet(DLTB).to_crs(UTM)
    out = out.reset_index(drop=True)
    rp = gpd.GeoDataFrame({"i": out.index}, geometry=out.geometry.representative_point(), crs=UTM)
    in_idx = gpd.sjoin(rp, dl[["geometry"]], predicate="within", how="inner")["i"].unique()
    out = out.loc[in_idx].reset_index(drop=True)
    out = out[~out.geometry.is_empty & out.geometry.notna()].explode(index_parts=False)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    print(f"[finalize] 裁县界(质心法): 保留 {len(out)} ({time.time()-t0:.0f}s)", flush=True)
    out["label_en"] = out["label"].map(EN); out["rgb_hex"] = out["label"].map(HEX)
    out["area_m2"] = out.geometry.area.round(1); out.insert(0, "gid", range(1, len(out) + 1))
    out.to_crs("EPSG:4326").to_parquet(OUT)
    print(f"[finalize] FINAL {len(out)} polys -> {OUT} ({time.time()-t0:.0f}s)", flush=True)

    # 面积对账 vs 三调
    dl["k"] = dl["DLBM"].astype(str).str[:2].map(M).fillna("荒漠")
    D = dl.assign(a=dl.geometry.area / 1e6).groupby("k")["a"].sum()
    P = out.assign(a=out.geometry.area / 1e6).groupby("label")["a"].sum()
    cls = ["耕地", "园地", "林地", "草地", "建筑", "水体", "荒漠"]
    T = pd.DataFrame({"TRUTH": D, "PRED": P}).reindex(cls).fillna(0)
    T["T%"] = (T["TRUTH"] / T["TRUTH"].sum() * 100).round(1)
    T["P%"] = (T["PRED"] / T["PRED"].sum() * 100).round(1)
    T["dpp"] = (T["P%"] - T["T%"]).round(1)
    print(T.round(1).to_string(), flush=True)
    print(f"[finalize] total PRED {T['PRED'].sum():.0f} vs TRUTH {T['TRUTH'].sum():.0f} km2", flush=True)

    # 出图: 全县预览 + 梯田放大(南部黄土丘陵)
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
            s.plot(ax=ax, facecolor=h, edgecolor="0.5", linewidth=0.05)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Yuzhong FINAL — enhance6(梯田) + Hann + dist-peak + 分幅缝合 + 平滑, %d polys" % len(g4), fontsize=13)
    lg = ax.legend(handles=[Patch(facecolor=h, label=c) for c, h in HEX.items()], loc="lower left", fontsize=12)
    for t in lg.get_texts():
        t.set_fontproperties(font)
    plt.savefig(f"/mnt/sda/zf/landform/results/yuzhong_{TAG}_preview.png", dpi=120, bbox_inches="tight", facecolor="white")
    # 梯田放大(南部丘陵带, 显示边界跟等高线)
    W = (104.18, 35.78, 104.30, 35.88)
    z = g4.cx[W[0]:W[2], W[1]:W[3]]
    fig, ax = plt.subplots(figsize=(15, 12))
    z.plot(ax=ax, color=z["rgb_hex"].values, edgecolor="black", linewidth=0.3)
    ax.set_xlim(W[0], W[2]); ax.set_ylim(W[1], W[3]); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("terrace zoom (梯田边界跟等高线?)", fontsize=12)
    plt.savefig(f"/mnt/sda/zf/landform/results/yuzhong_{TAG}_terrace.png", dpi=140, bbox_inches="tight", facecolor="white")
    print("SAVED_OK", flush=True)


if __name__ == "__main__":
    main()
