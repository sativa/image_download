import geopandas as gpd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/mnt/sda/zf/landform/results"
c1m = gpd.read_parquet(f"{R}/yuzhong_c1m_region.parquet")
tif = gpd.read_parquet(f"{R}/yuzhong_tif_region.parquet")

fig, axes = plt.subplots(1, 2, figsize=(22, 11))
for ax, g, t in [(axes[0], c1m, f"Route1 download c_1m (~1m): {len(c1m)} parcels"),
                 (axes[1], tif, f"Route2 local (~2m): {len(tif)} parcels")]:
    g.plot(color=g["rgb_hex"].tolist(), ax=ax, linewidth=0)
    ax.set_title(t, fontsize=14); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.savefig(f"{R}/yuzhong_preview.png", dpi=110, bbox_inches="tight"); plt.close()
print("① 预览 -> yuzhong_preview.png")

def crop_ha_by_cell(g):
    c = g[g["class_id"].isin([1, 2])]
    return c.groupby("cell")["area_m2"].sum() / 1e4
a, b = crop_ha_by_cell(c1m), crop_ha_by_cell(tif)
common = sorted(set(a.index) & set(b.index))
print(f"② 两路线公共 cell = {len(common)}")
if len(common) > 2:
    av, bv = a.reindex(common).fillna(0).values, b.reindex(common).fillna(0).values
    r = np.corrcoef(av, bv)[0, 1]
    print(f"   逐cell耕地面积相关 r = {r:.3f}")
    print(f"   公共cell耕地: 路线1 {av.sum():.0f} ha vs 路线2 {bv.sum():.0f} ha (差 {abs(av.sum()-bv.sum())/av.sum()*100:.1f}%)")
for tag, g in [("路线1 c_1m", c1m), ("路线2 tif", tif)]:
    d = g.groupby("label_en")["area_m2"].sum().div(1e6).round(1).sort_values(ascending=False)
    print(f"   {tag} 各类 km²: {d.to_dict()}")
