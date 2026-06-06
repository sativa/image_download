import geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g = gpd.read_parquet("/mnt/sda/zf/landform/results/yuzhong_continuous_region.parquet")
fig, ax = plt.subplots(figsize=(14, 14))
g.plot(color=g["rgb_hex"].tolist(), ax=ax, linewidth=0.04, edgecolor="white")
ax.set_title(f"Yuzhong continuous ~12x12km: {len(g)} parcels (dist-peak, 7-class)", fontsize=13)
ax.set_aspect("equal"); ax.axis("off")
plt.savefig("/mnt/sda/zf/landform/results/yuzhong_continuous_preview.png", dpi=140, bbox_inches="tight"); plt.close()
print("preview saved,", len(g), "parcels")
