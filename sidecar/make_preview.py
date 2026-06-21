"""通用 landuse 预览图:左=全市(按 rgb_hex 着色+图例含逐类面积),右=局部放大(白描边看边界曲线)。
用法: python make_preview.py <parquet> <out.png> "<title>" [zoom_cx zoom_cy](4326度,可选,默认取范围中心)
"""
import sys
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

P = sys.argv[1]
OUT = sys.argv[2]
TITLE = sys.argv[3] if len(sys.argv) > 3 else "landuse"

g = gpd.read_parquet(P)
gu = g.to_crs(32649)
area_km2 = gu.area.groupby(g["class_id"].values).sum() / 1e6
seen = g.drop_duplicates("class_id").sort_values("class_id")

fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=150)
# 左:全市
ax = axes[0]
for _, r in seen.iterrows():
    g[g["class_id"] == r["class_id"]].plot(ax=ax, color=r["rgb_hex"], linewidth=0, antialiased=False)
ax.set_axis_off(); ax.set_aspect("equal")
ax.set_title("%s (%d parcels, %.0f km2)" % (TITLE, len(g), gu.area.sum() / 1e6), fontsize=14)
handles = [Patch(facecolor=r["rgb_hex"], edgecolor="none",
                 label="%s  (%.0f km2)" % (r["label_en"], area_km2.get(r["class_id"], 0.0)))
           for _, r in seen.iterrows()]
ax.legend(handles=handles, loc="lower left", fontsize=10, title="class (area)", framealpha=0.92)
# 右:局部放大(白描边 -> 看边界是曲线还是折线 + 块边有无接缝)
minx, miny, maxx, maxy = g.total_bounds
cx = float(sys.argv[4]) if len(sys.argv) > 4 else (minx + maxx) / 2
cy = float(sys.argv[5]) if len(sys.argv) > 5 else (miny + maxy) / 2
d = 0.022   # ~2.2km(度)
ax2 = axes[1]
sub = g.cx[cx - d:cx + d, cy - d:cy + d]
for _, r in seen.iterrows():
    s = sub[sub["class_id"] == r["class_id"]]
    if len(s):
        s.plot(ax=ax2, color=r["rgb_hex"], linewidth=0.45, edgecolor="white", antialiased=True)
ax2.set_aspect("equal"); ax2.set_xlim(cx - d, cx + d); ax2.set_ylim(cy - d, cy + d)
ax2.set_title("zoom ~4.4km (white edges: curved vs polygonal, seam check)", fontsize=14)

fig.savefig(OUT, bbox_inches="tight", dpi=150, facecolor="white")
print("WROTE %s | parcels=%d area=%.1fkm2" % (OUT, len(g), gu.area.sum() / 1e6))
