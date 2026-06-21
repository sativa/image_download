"""make_changzhi_preview — 长治市 landuse 全市预览图(按类 rgb_hex 着色 + 图例含逐类面积)。"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

P = "/mnt/sda/zf/landform/results/changzhi_FINAL.parquet"
OUT = "/mnt/sda/zf/landform/results/changzhi_preview.png"

g = gpd.read_parquet(P)
gu = g.to_crs(32649)
area_km2 = gu.area.groupby(g["class_id"].values).sum() / 1e6

fig, ax = plt.subplots(figsize=(16, 14), dpi=160)
seen = g.drop_duplicates("class_id").sort_values("class_id")
for _, r in seen.iterrows():
    g[g["class_id"] == r["class_id"]].plot(ax=ax, color=r["rgb_hex"], linewidth=0, antialiased=False)
ax.set_axis_off()
ax.set_aspect("equal")
ax.set_title("Changzhi City — 1m parcel landuse (%d parcels, %.0f km2, seamless)" % (len(g), gu.area.sum() / 1e6),
             fontsize=15)
handles = [Patch(facecolor=r["rgb_hex"], edgecolor="none",
                 label="%s  (%.0f km2)" % (r["label_en"], area_km2.get(r["class_id"], 0.0)))
           for _, r in seen.iterrows()]
ax.legend(handles=handles, loc="lower left", fontsize=11, title="class (area)", framealpha=0.92)
fig.savefig(OUT, bbox_inches="tight", dpi=160, facecolor="white")
print("WROTE %s" % OUT)
