"""recon_changzhi — 把分幅(yz_blocks 方案B)产物 changzhi_FINAL.parquet 修成干净无缝成品。
分幅缺口:bbox-core 质心归属在块边重复计数(实测 ~14% 面积重复)+ 未裁市界。
本脚本(复用现有件,零改核心):① 裁精确市界(质心-in-boundary,快;line ec1b664 同法)
② pp.resolve_overlaps 去块边重复(较小者让出重叠区,exact 零重叠)③ postproc.run_postproc 标准收尾。
"""
import sys, time, collections
sys.path.insert(0, "/home/ps/landform/sidecar")
import geopandas as gpd
import parcel_pipeline as pp
import postproc

CLASSES = [(1, "耕地", "cropland", (60, 180, 75)), (2, "园地", "orchard", (170, 255, 90)),
           (3, "林地", "forest", (0, 100, 0)), (4, "草地", "grassland", (190, 220, 100)),
           (5, "水体", "water", (0, 130, 200)), (6, "建筑", "built-up", (230, 25, 75)),
           (7, "荒漠", "bare", (170, 140, 100))]
UTM = "EPSG:32649"
IN = "/mnt/sda/zf/landform/results/changzhi_FINAL.parquet"
OUT = "/mnt/sda/zf/landform/results/changzhi_RECON.parquet"
BND = "/mnt/sda/zf/landform/data/changzhi_boundary.geojson"

t0 = time.time()
def el():
    return "%.0fs" % (time.time() - t0)

g = gpd.read_parquet(IN).to_crs(UTM)
b = gpd.read_file(BND).to_crs(UTM)
bgeom = b.union_all()
print("[recon] loaded %d parcels (%s)" % (len(g), el()), flush=True)

# 1. 裁市界:质心 in boundary(sjoin 加速)
cent = gpd.GeoDataFrame(geometry=g.geometry.centroid, crs=g.crs)
keep = gpd.sjoin(cent, gpd.GeoDataFrame(geometry=[bgeom], crs=g.crs),
                 predicate="within", how="inner").index.unique()
g = g.loc[keep][["class_id", "geometry"]].reset_index(drop=True)
print("[recon] centroid-clip -> %d parcels, area %.1f km2 (%s)" % (len(g), g.area.sum() / 1e6, el()), flush=True)

# 2. 去块边重复
g2 = pp.resolve_overlaps(g).to_crs(UTM)
print("[recon] resolve_overlaps -> %d parcels, area %.1f km2 (%s)" % (len(g2), g2.area.sum() / 1e6, el()), flush=True)

# 3. 标准收尾(无缝输入 skip_gaps,清残余 sliver/微洞 + 标准化)
clean, rep = postproc.run_postproc(g2, CLASSES, boundary=bgeom, utm=UTM, skip_gaps=True)
clean.to_parquet(OUT)
cu = clean.to_crs(UTM)
print("[recon] DONE %d parcels, total %.1f km2 -> %s (%s)" % (len(clean), cu.area.sum() / 1e6, OUT, el()), flush=True)
ar = cu.assign(_l=clean["label"].values).groupby("_l").apply(lambda d: round(d.geometry.area.sum() / 1e6, 1))
print("[recon] per-class km2:", dict(ar), flush=True)
print("RECON_EXIT=0", flush=True)
