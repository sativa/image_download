"""Qualitative case-study panels for the paper: for a few terrace test cells, render
RGB (Esri) | predicted cropland | DLTB ground truth, using the best final model
(DINOv3-Sat + FreqFusion + GDLX, last ckpt). Saves a multi-row PNG.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from parcel_eval import cropland_prob, load_county
from train_dino_1m_v2 import load_ndvi_full
from train_dino_1m_v3 import DinoV3FreqUNet
from transformers import AutoModel

D = Path("/mnt/sda/zf/landform/data/c_1m_terrace2")
DLTB = "/home/ps/landform/data/v11_dltb"
BK = "/home/ps/landform/dinov3/dinov3-vitl16-sat493m"
CKPT = "/mnt/sda/zf/landform/results/dino_1m_v3_gdlxff_max/last.pt"
OUT = "/mnt/sda/zf/landform/results/figures/case_study_terrace.png"
# hand-picked terrace cells (dense, visually clear)
CELLS = ["621124_521301769", "622922_518201762", "620522_528001738"]

dev = "cuda:0"
d3 = AutoModel.from_pretrained(BK, local_files_only=True)
model = DinoV3FreqUNet(d3, num_classes=3, in_channels=11, unfreeze_last_n=4).to(dev)
sd = torch.load(CKPT, map_location=dev, weights_only=True)
model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()
                       and model.state_dict()[k].shape == v.shape}, strict=False)
model.eval()
print("model loaded", flush=True)

cache = {}
rows = [c for c in CELLS if (D / f"{c}.npz").exists()][:3]
fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.7 * len(rows)))
if len(rows) == 1: axes = axes[None, :]
cmap_gt = ListedColormap(["#00000000", "#2ca02c", "#d62728"])  # 0 nodata,1 crop,2 noncrop

for r, n in enumerate(rows):
    z = np.load(D / f"{n}.npz"); x6 = z["x6"]; lbl = z["label"]; bbox = z["bbox"]
    _, H, W = x6.shape
    ndvi = load_ndvi_full(n, H, W)
    if ndvi is None: ndvi = np.zeros((5, H, W), np.float32)
    prob = cropland_prob(model, x6, dev, ndvi=ndvi)
    rgb = np.transpose(x6[:3], (1, 2, 0)).astype(np.uint8)   # Esri RGB
    # DLTB parcel raster for the GT panel
    g = load_county(DLTB, n.split("_")[0], cache)
    tr = from_bounds(*[float(v) for v in bbox], W, H)
    idx = list(g.sindex.intersection(tuple(float(v) for v in bbox)))
    gt = np.zeros((H, W), np.uint8)
    if idx:
        cb = shp_box(*bbox); sub = g.iloc[idx]
        shapes = [(geom, 1 if int(c) in (1, 2) else 2) for geom, c in zip(sub.geometry, sub["cid"]) if geom.intersects(cb)]
        if shapes: gt = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
    axes[r, 0].imshow(rgb); axes[r, 0].set_ylabel(n, fontsize=9)
    axes[r, 1].imshow(rgb); axes[r, 1].imshow(prob, cmap="RdYlGn", alpha=0.55, vmin=0, vmax=1)
    axes[r, 2].imshow(rgb); axes[r, 2].imshow(gt, cmap=cmap_gt, alpha=0.55, vmin=0, vmax=2)
    for c in range(3):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    if r == 0:
        axes[0, 0].set_title("(a) 1 m RGB (Esri)", fontsize=11)
        axes[0, 1].set_title("(b) Predicted cropland probability", fontsize=11)
        axes[0, 2].set_title("(c) DLTB ground truth (green=cropland)", fontsize=11)
plt.tight_layout()
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=220, bbox_inches="tight")
print("saved", OUT, flush=True)
