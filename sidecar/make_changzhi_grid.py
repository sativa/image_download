"""make_changzhi_grid — 从长治市边界(缓冲)生成连续全市下载网格 + 精确裁切边界。

输入: 长治边界缓冲.shp(EPSG:4526 米制,1 feature,带 BUFF_DIST = 缓冲距离)。
输出:
  changzhi_full_regions.json  —— downloader(imagery-downloader batch --regions)用的连续 0.02° 网格,
                                 [{"county","idx","bbox":[lon0,lat0,lon1,lat1]}, ...],∩ 缓冲边界(保证全市边缘全覆盖)。
  changzhi_boundary.geojson   —— 精确长治市边界(缓冲负缓冲还原),wgs84,管线 --boundary 最终裁切用。
"""
import json
import sys
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

DATA = "/mnt/sda/zf/landform/data"
BUF_SHP = DATA + "/长治边界缓冲.shp"
OUT_REGIONS = DATA + "/changzhi_full_regions.json"
OUT_BOUNDARY = DATA + "/changzhi_boundary.geojson"
STEP = 0.02   # ~2km cell,与神池/采样同口径

b = gpd.read_file(BUF_SHP)
buff_dist = float(b["BUFF_DIST"].iloc[0]) if "BUFF_DIST" in b.columns else 0.0
print("[grid] buffered boundary crs=%s BUFF_DIST=%s" % (b.crs, buff_dist), flush=True)

# 精确边界 = 缓冲负缓冲还原(在米制 EPSG:4526 直接做)
if buff_dist:
    exact_m = b.geometry.buffer(-abs(buff_dist))
    exact = gpd.GeoDataFrame(geometry=exact_m, crs=b.crs)
    exact = exact[~exact.geometry.is_empty]
else:
    exact = b
exact.to_crs(4326).to_file(OUT_BOUNDARY, driver="GeoJSON")
print("[grid] wrote exact boundary -> %s (area km2=%.1f)"
      % (OUT_BOUNDARY, exact.to_crs(32649).area.sum() / 1e6), flush=True)

# 连续网格 ∩ 缓冲边界(用缓冲版保证边缘全覆盖)
bw = b.to_crs(4326)
geom = bw.geometry.union_all()
pg = prep(geom)
minx, miny, maxx, maxy = geom.bounds
xs = np.arange(minx, maxx, STEP)
ys = np.arange(miny, maxy, STEP)
cells = []
for r, y in enumerate(ys):
    for c, x in enumerate(xs):
        cb = box(x, y, x + STEP, y + STEP)
        if pg.intersects(cb):
            cells.append({"county": "changzhi_%d" % r, "idx": int(c),
                          "bbox": [float(x), float(y), float(min(x + STEP, maxx)), float(min(y + STEP, maxy))]})
json.dump(cells, open(OUT_REGIONS, "w"))
print("[grid] grid cells=%d (over %d x %d lattice) -> %s"
      % (len(cells), len(xs), len(ys), OUT_REGIONS), flush=True)
print("[grid] bbox wgs84:", [round(z, 4) for z in (minx, miny, maxx, maxy)], flush=True)
