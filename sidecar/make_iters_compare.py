"""同一小窗口出 Chaikin iters=0/1/2/3 对比(tol=15),看平滑强度 vs 顶点数,选合适档位。"""
import sys, pickle
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import get_num_coordinates as gnc
from affine import Affine
import parcel_pipeline as pp

R = "/mnt/sda/zf/landform/results"
H, W, H4, W4, tr4, crs = pp._read_grid_meta(R + "/changzhi_mosaic.vrt", 4)
idmap = np.load(R + "/changzhi_inter/idmap.npy")
cls_of = pickle.load(open(R + "/changzhi_inter/cls_of.pkl", "rb"))  # 可信:本会话自写

r0, c0, sz = 16500, 18500, 1800
sub = idmap[r0:r0 + sz, c0:c0 + sz]
ids = [int(i) for i in np.unique(sub) if i > 0]
subcls = {i: cls_of[i] for i in ids if i in cls_of}
subtr = tr4 * Affine.translation(c0, r0)
raw = pp.vectorize_idmap(sub, subcls, subtr, crs)
print("window parcels:", len(subcls), flush=True)

CMAP = {1: (0.24, 0.71, 0.29), 2: (0.67, 1, 0.35), 3: (0, 0.39, 0), 4: (0.75, 0.86, 0.39),
        5: (0, 0.51, 0.78), 6: (0.9, 0.1, 0.29), 7: (0.67, 0.55, 0.39)}
fig, axes = plt.subplots(1, 4, figsize=(26, 7), dpi=140)
for ax, it in zip(axes, [0, 1, 2, 3]):
    sm = pp.smooth_coverage(raw, tol=15, iters=it)
    v = gnc(sm.geometry.values)
    for cid, col in CMAP.items():
        s = sm[sm["class_id"] == cid]
        if len(s):
            s.plot(ax=ax, color=col, linewidth=0.6, edgecolor="white", antialiased=True)
    ax.set_title("iters=%d  |  mean %.0f 顶点/地块%s" % (it, v.mean(), "  (=FINAL折线)" if it == 0 else ""), fontsize=13)
    ax.set_aspect("equal"); ax.set_axis_off()
fig.suptitle("Chaikin 平滑强度对比(同窗口, tol=15):iters 越高越圆、顶点越多(体积越大)", fontsize=15)
fig.savefig(R + "/changzhi_iters_compare.png", bbox_inches="tight", dpi=140, facecolor="white")
print("WROTE " + R + "/changzhi_iters_compare.png", flush=True)
