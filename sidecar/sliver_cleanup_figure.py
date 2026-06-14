"""Before/after comparison figure on a SPARSE window (most w<0.5m slivers).
Left = before clean (floating dangling lines), right = after clean. Boundary view: orange on white.
Reports real-field (>100m2) retention in the window.
"""
import sys, os
SIDECAR = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar"
sys.path.insert(0, SIDECAR); os.chdir(SIDECAR)
import numpy as np
import geopandas as gpd
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEFORE = "yuzhong_product/yuzhong_SMOOTH3_chaikin3.parquet"
AFTER  = "yuzhong_product/yuzhong_SMOOTH3_chaikin3_clean.parquet"
UTM = "EPSG:32648"

def find_window(path, half_m=600.0):
    g = gpd.read_parquet(path).to_crs(UTM)
    geoms = g.geometry.values
    a = shapely.area(geoms); p = shapely.length(geoms)
    w = np.where(p > 0, a / p, 0.0)
    fine = (w < 0.5)
    rp = shapely.point_on_surface(geoms[fine])
    x = shapely.get_x(rp); y = shapely.get_y(rp)
    bs = 2 * half_m
    from collections import Counter
    c = Counter((int(xx // bs), int(yy // bs)) for xx, yy in zip(x, y))
    (bx, by), cnt = c.most_common(1)[0]
    cx = (bx + 0.5) * bs; cy = (by + 0.5) * bs
    print(f"densest w<0.5 window center UTM ({cx:.0f},{cy:.0f}) with {cnt} fine slivers", flush=True)
    return cx, cy

def load_win(path, cx, cy, half):
    g = gpd.read_parquet(path).to_crs(UTM)
    sub = g.cx[cx-half:cx+half, cy-half:cy+half].copy()
    sub = gpd.clip(sub, (cx-half, cy-half, cx+half, cy+half))
    return sub[~sub.geometry.is_empty & sub.geometry.notna()]

def plot_b(ax, gdf, color="#ff7f0e", lw=0.7):
    for geom in gdf.geometry:
        if geom is None or geom.is_empty: continue
        for poly in getattr(geom, "geoms", [geom]):
            if poly.geom_type != "Polygon": continue
            xs, ys = poly.exterior.xy; ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")
            for r in poly.interiors:
                xs, ys = r.xy; ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")

def cnt_fine(gdf):
    geoms = gdf.geometry.values
    if len(geoms) == 0: return 0, 0
    a = shapely.area(geoms); p = shapely.length(geoms)
    w = np.where(p > 0, a / p, 0.0)
    return int((w < 0.5).sum()), int((a > 100).sum())

def main():
    half = 600.0
    cx, cy = find_window(BEFORE, half_m=half)
    gb = load_win(BEFORE, cx, cy, half); ga = load_win(AFTER, cx, cy, half)
    fb, rb = cnt_fine(gb); fa, ra = cnt_fine(ga)
    print(f"BEFORE win n={len(gb)} fine_w<0.5={fb} real>100m2={rb}", flush=True)
    print(f"AFTER  win n={len(ga)} fine_w<0.5={fa} real>100m2={ra}", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=170)
    for ax, gdf, title in [
        (axes[0], gb, f"BEFORE  (fine slivers w<0.5m: {fb})  - floating dangling lines"),
        (axes[1], ga, f"AFTER clean  (fine slivers w<0.5m: {fa})  - artifacts removed")]:
        ax.set_facecolor("white"); plot_b(ax, gdf)
        ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half); ax.set_aspect("equal")
        ax.set_title(title, fontsize=15); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#cccccc")
    fig.suptitle(f"Yuzhong boundaries - sliver/dangling-line cleanup (sparse 1.2km window UTM48N {cx:.0f},{cy:.0f}; real fields>100m2 {rb}->{ra})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    o1="/tmp/yuzhong_sliver_cleanup_compare.png"; o2=SIDECAR+"/yuzhong_product/yuzhong_sliver_cleanup_compare.png"
    fig.savefig(o1, dpi=170); fig.savefig(o2, dpi=170)
    print("saved", o1, o2, flush=True); print("FIG_DONE", flush=True)

if __name__ == "__main__":
    main()
