"""省级方案验证(用长治全局 idmap 当案例):测"分块矢量化 + 按全局 instance ID dissolve"能否无缝重建全局。
省级方案核心 = 分区(idmap 标签)全局一致 → 分块各自矢量化(块边切碎)→ 按全局 ID dissolve 把碎片并回 → 无缝。
本测试:把长治全局 idmap 切块、逐块 vectorize(块边截断成 parcel_id 相同的多片)→ dissolve by parcel_id →
对比"直接全局 vectorize"的面积/实例数。Δ≈0 ⟹ 分块-dissolve 重建=全局,合并原理可行。
注:仅验证"合并原理"(用现成全局 idmap);未验证"idmap 太大装不下→粗标签/标签传播"那半(长治 idmap 装得下)。
"""
import sys, time, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import geopandas as gpd
import rasterio.features
from shapely.geometry import shape as _shape
from affine import Affine
import parcel_pipeline as pp

INTER = "/mnt/sda/zf/landform/results/changzhi_inter"
VRT = "/mnt/sda/zf/landform/results/changzhi_mosaic.vrt"
UTM = "EPSG:32649"; TILE = 10000


def _vec_sub(sub, subtr, cls_of):
    out = []
    for geom, val in rasterio.features.shapes(sub.astype(np.int32), mask=sub > 0, connectivity=8, transform=subtr):
        v = int(val); c = cls_of.get(v)
        if not c:
            continue
        g = _shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        out.append({"parcel_id": v, "class_id": int(c), "geometry": g})
    return out


def main():
    t0 = time.time()
    def el():
        return "%.0fmin" % ((time.time() - t0) / 60)
    H, W, H4, W4, tr4, crs = pp._read_grid_meta(VRT, 4)
    idmap = np.load(INTER + "/idmap.npy")
    cls_of = pickle.load(open(INTER + "/cls_of.pkl", "rb"))  # 可信:本会话自写 int->int
    # 分块矢量化(块边切碎,parcel_id=全局标签)
    pieces = []; nt = 0
    for r0 in range(0, H4, TILE):
        for c0 in range(0, W4, TILE):
            sub = idmap[r0:r0 + TILE, c0:c0 + TILE]
            if not (sub > 0).any():
                continue
            nt += 1
            subtr = tr4 * Affine.translation(c0, r0)
            pieces.extend(_vec_sub(sub, subtr, cls_of))
    print("[prov] %d tiles -> %d pieces(含块边切碎) (%s)" % (nt, len(pieces), el()), flush=True)
    gdf = gpd.GeoDataFrame(pieces, crs=crs)
    n_multi = int((gdf.groupby("parcel_id").size() > 1).sum())
    print("[prov] 跨块被切的实例数: %d (%s)" % (n_multi, el()), flush=True)
    # 按全局 ID dissolve -> 并回碎片
    diss = gdf.dissolve(by="parcel_id", aggfunc="first")
    du = diss.to_crs(UTM)
    print("[prov] dissolve -> %d 全局实例, 面积 %.2f km2 (%s)" % (len(diss), du.area.sum() / 1e6, el()), flush=True)
    # 对比直接全局 vectorize
    rawg = pp.vectorize_idmap(idmap, cls_of, tr4, crs)
    rgd = rawg.dissolve(by="parcel_id", aggfunc="first")
    rgu = rgd.to_crs(UTM)
    print("[prov] 直接全局 -> %d 实例, 面积 %.2f km2 (%s)" % (len(rgd), rgu.area.sum() / 1e6, el()), flush=True)
    print("[prov] ★ 面积Δ=%.4f km2 | 实例数Δ=%d (≈0 ⟹ 分块-dissolve 无缝重建=全局, 合并原理可行)"
          % (abs(du.area.sum() - rgu.area.sum()) / 1e6, len(diss) - len(rgd)), flush=True)
    print("PROV_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
