import geopandas as gpd
for tag, f in [("路线1 c_1m(1m)", "/mnt/sda/zf/landform/results/yuzhong_c1m_region.parquet"),
               ("路线2 本地(2m)", "/mnt/sda/zf/landform/results/yuzhong_tif_region.parquet")]:
    g = gpd.read_parquet(f)
    crop = g[g["class_id"].isin([1, 2])]
    print(f"{tag}: {len(g)} 地块 | CRS {g.crs} | 耕地+园地 {len(crop)} 块 "
          f"{round(crop['area_m2'].sum()/1e6,1)} km² | 全类 {round(g['area_m2'].sum()/1e6,1)} km² | "
          f"列 {list(g.columns)}")
