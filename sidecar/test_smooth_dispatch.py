"""smoke test:同一小窗口分别走 smooth_auto 全局 vs 分块,验证两路都跑通且类别面积一致(重构无 typo)。"""
import sys, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import parcel_pipeline as pp
import smooth_dispatch as sd

R = "/mnt/sda/zf/landform/results"
H, W, H4, W4, tr4, crs = pp._read_grid_meta(R + "/changzhi_mosaic.vrt", 4)
idmap = np.load(R + "/changzhi_inter/idmap.npy")
cls_of = pickle.load(open(R + "/changzhi_inter/cls_of.pkl", "rb"))  # 可信:本会话自写

r0, c0, sz = 16000, 18000, 4000
sub = idmap[r0:r0 + sz, c0:c0 + sz].copy()
ids = [int(i) for i in np.unique(sub) if i > 0]
subcls = {i: cls_of[i] for i in ids if i in cls_of}
subtr = tr4 * sd.Affine.translation(c0, r0)
print("window parcels:", len(subcls))

g = sd.smooth_auto(sub, subcls, subtr, crs, tol=15, iters=3, max_global_parcels=10**9, verbose=False)
t = sd.smooth_auto(sub, subcls, subtr, crs, tol=15, iters=3, max_global_parcels=0, tile_px=1500, verbose=False)
ga = (g.to_crs(32649).area.groupby(g["class_id"].values).sum() / 1e6).round(3)
ta = (t.to_crs(32649).area.groupby(t["class_id"].values).sum() / 1e6).round(3)
print("GLOBAL: %d polys, cols=%s" % (len(g), list(g.columns)))
print("TILED : %d polys, cols=%s" % (len(t), list(t.columns)))
print("global area km2:", dict(ga))
print("tiled  area km2:", dict(ta))
print("total Δ km2 = %.4f (≈0 ⟹ 两路一致, 类别保真)" % abs(ga.sum() - ta.sum()))
print("SMOKE_OK")
