"""对比图: 旧 yuzhong_FINAL(草地/林地边折角, 老 giant 判据) vs 新 v2b(草地/林地边曲线, 新线状判据)。
取若干"田块(耕地)挨草地/林地"的窗, boundary 橙线白底 dpi>=150。目视新版田块↔草地/林地边更平滑。"""
import sys, time
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import geopandas as gpd
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/mnt/sda/zf/landform/results"
OLD = f"{RES}/yuzhong_FINAL.parquet"
NEW = f"{RES}/yuzhong_FINAL_v2b.parquet"
OUTDIR = "/tmp"


def boundary_lines(gdf):
    return gdf.geometry.boundary


def find_windows(gnew, n=4, half_m=180.0):
    """找耕地与草地/林地相邻的位置: 取耕地图斑质心, 其 ~120m 内既有草地又有林地/草地的窗。
    在 UTM 下选, 返回若干 (cx, cy) 中心(UTM 米)。"""
    gu = gnew.to_crs("EPSG:32648")
    crop = gu[gu["label"] == "耕地"].reset_index(drop=True)
    other = gu[gu["label"].isin(["草地", "林地"])].reset_index(drop=True)
    if len(crop) == 0 or len(other) == 0:
        return []
    otree = shapely.STRtree(other.geometry.values)
    # 选面积适中的耕地(避免巨块/碎块), 且邻接草地/林地
    crop = crop[(crop["area_m2"] > 3000) & (crop["area_m2"] < 60000)].reset_index(drop=True)
    cents = crop.geometry.centroid
    rng = np.random.default_rng(42)
    order = rng.permutation(len(crop))
    wins = []
    for idx in order:
        c = cents.iloc[idx]
        cx, cy = c.x, c.y
        win = shapely.box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
        hit = otree.query(win, predicate="intersects")
        if len(hit) >= 1:
            # 确认窗内草地/林地确有相当面积(不是只蹭到一角)
            oa = sum(other.geometry.values[h].intersection(win).area for h in hit)
            if oa > (half_m * half_m * 0.15):
                wins.append((cx, cy))
        if len(wins) >= n:
            break
    return wins


def plot_pair(gold, gnew, wins, half_m=180.0):
    nrow = len(wins)
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 5.6 * nrow))
    if nrow == 1:
        axes = axes.reshape(1, 2)
    gold_u = gold.to_crs("EPSG:32648")
    gnew_u = gnew.to_crs("EPSG:32648")
    for r, (cx, cy) in enumerate(wins):
        win = shapely.box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
        for c, (g, title) in enumerate([(gold_u, "OLD yuzhong_FINAL (giant=verts only)"),
                                        (gnew_u, "NEW v2b (giant=verts AND linear)")]):
            ax = axes[r][c]
            sub = g[g.geometry.intersects(win)]
            for geom, lab in zip(sub.geometry.values, sub["label"].values):
                gi = geom.intersection(win)
                if gi.is_empty:
                    continue
                col = "tab:orange"
                # 耕地边突出深橙, 草地/林地边稍浅, 其余灰
                lw = 1.4
                if lab in ("草地", "林地"):
                    col = "tab:green"
                elif lab == "耕地":
                    col = "tab:orange"
                else:
                    col = "0.6"; lw = 0.8
                bnd = gi.boundary
                for ls in (bnd.geoms if bnd.geom_type.startswith("Multi") else [bnd]):
                    try:
                        xs, ys = ls.xy
                        ax.plot(xs, ys, color=col, linewidth=lw, solid_capstyle="round")
                    except Exception:
                        pass
            ax.set_xlim(cx - half_m, cx + half_m); ax.set_ylim(cy - half_m, cy + half_m)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("%s\nwin#%d (orange=耕地 green=草地/林地)" % (title, r + 1), fontsize=9)
    plt.tight_layout()
    out = f"{OUTDIR}/yz_v2b_smooth_compare.png"
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("[fig] saved %s" % out, flush=True)
    return out


def main():
    t0 = time.time()
    gold = gpd.read_parquet(OLD)
    gnew = gpd.read_parquet(NEW)
    print("[fig] OLD %d polys | NEW %d polys (%.0fs)" % (len(gold), len(gnew), time.time() - t0), flush=True)
    wins = find_windows(gnew, n=4, half_m=180.0)
    print("[fig] %d windows (UTM): %s" % (len(wins), [(round(x), round(y)) for x, y in wins]), flush=True)
    if not wins:
        print("[fig] NO windows found", flush=True)
        return
    plot_pair(gold, gnew, wins, half_m=180.0)
    print("FIG_OK", flush=True)


if __name__ == "__main__":
    main()
