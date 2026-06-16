"""快对比图 v3: 用 fig1 已找到的好窗(耕地挨草地/林地)做紧凑放大(half=110m), ASCII 标题免CJK方框。
左 OLD yuzhong_FINAL(草地/林地边折角) vs 右 v2b(草地/林地边曲线)。橙=耕地边 绿=草地/林地边 灰=其他。"""
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
OUT = "/tmp/yz_v2b_smooth_compare3.png"
# fig1 已确认的耕地↔草地/林地窗心(UTM 32648); 各取两档放大
WINS = [(440462, 3955726), (441542, 3973170), (414240, 3982431), (445874, 4006838)]
HALF = 110.0


def plot_pair(gold, gnew):
    nrow = len(WINS)
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 5.2 * nrow))
    if nrow == 1:
        axes = axes.reshape(1, 2)
    gold_u = gold.to_crs("EPSG:32648"); gnew_u = gnew.to_crs("EPSG:32648")
    for r, (cx, cy) in enumerate(WINS):
        win = shapely.box(cx - HALF, cy - HALF, cx + HALF, cy + HALF)
        for c, (g, title) in enumerate([(gold_u, "OLD FINAL (giant=verts-only: grass/forest edge FOLDED)"),
                                        (gnew_u, "NEW v2b (giant=verts AND linear: grass/forest edge CURVED)")]):
            ax = axes[r][c]
            sub = g[g.geometry.intersects(win)]
            for geom, lab in zip(sub.geometry.values, sub["label"].values):
                gi = geom.intersection(win)
                if gi.is_empty:
                    continue
                if lab in ("草地", "林地"):
                    col, lw = "tab:green", 1.8
                elif lab == "耕地":
                    col, lw = "tab:orange", 1.8
                else:
                    col, lw = "0.75", 0.7
                bnd = gi.boundary
                for ls in (bnd.geoms if bnd.geom_type.startswith("Multi") else [bnd]):
                    try:
                        xs, ys = ls.xy
                        ax.plot(xs, ys, color=col, linewidth=lw, solid_capstyle="round")
                    except Exception:
                        pass
            ax.set_xlim(cx - HALF, cx + HALF); ax.set_ylim(cy - HALF, cy + HALF)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("%s\nwin#%d  orange=cropland  green=grass/forest" % (title, r + 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT, dpi=185, bbox_inches="tight", facecolor="white")
    print("[fig3] saved %s" % OUT, flush=True)


def main():
    t0 = time.time()
    gold = gpd.read_parquet(OLD); gnew = gpd.read_parquet(NEW)
    print("[fig3] OLD %d | NEW %d (%.0fs)" % (len(gold), len(gnew), time.time() - t0), flush=True)
    plot_pair(gold, gnew)
    print("FIG3_OK", flush=True)


if __name__ == "__main__":
    main()
