"""对比 FINAL(无Chaikin折线) vs tiled_smooth(Chaikin3) 的顶点数,定位体积差主因。"""
import geopandas as gpd
from shapely import get_num_coordinates as gnc

R = "/mnt/sda/zf/landform/results"
for name, p in [("FINAL  全局/无Chaikin(折线)", "changzhi_FINAL.parquet"),
                ("tiled  分块/Chaikin iters=3", "changzhi_tiled_smooth.parquet")]:
    g = gpd.read_parquet(R + "/" + p)
    v = gnc(g.geometry.values)
    print("%-30s: %7d 地块 | 总顶点 %10d | mean %6.1f | median %4d | max %6d"
          % (name, len(g), int(v.sum()), v.mean(), int(sorted(v)[len(v) // 2]), int(v.max())), flush=True)
