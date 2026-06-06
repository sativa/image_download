"""Parcel-level cropland PRODUCT demo (best model dino_1m_v2_smallw).

For a few Gansu test cells: run the model -> 1m cropland prob; take DLTB parcels; assign each parcel
its majority-vote class -> a classified parcel map. Renders [RGB | predicted cropland parcels | DLTB
ground-truth parcels] so visual agreement = the 0.92 area-F1 made tangible. Also writes a GeoJSON of
classified parcels (pred vs true) per cell -> a real GIS-openable deliverable.
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m_v2 import load_ndvi_full, DinoUNetBoundary
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from parcel_eval import load_county, cropland_prob

CMAP = ListedColormap(["#ffffff", "#3a9d23", "#d9c08a"])  # 0 nodata / 1 cropland(green) / 2 other(tan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m_v2_smallw/best.pt")
    p.add_argument("--multitemporal", action="store_true", default=True)
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/figures")
    p.add_argument("--cells", default="", help="comma names; default = spread across test set")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device; in_ch = 11 if a.multitemporal else 6

    from transformers import AutoModel
    d = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNetBoundary(DinoUNet5ch(d, num_classes=3, in_channels=in_ch, unfreeze_last_n=4)).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True)); model.eval()

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    cells = a.cells.split(",") if a.cells else te[:: max(1, len(te) // a.n)][:a.n]
    print(f"[demo] cells={cells}", flush=True)

    cache = {}
    fig, axes = plt.subplots(len(cells), 3, figsize=(13, 4.3 * len(cells)))
    if len(cells) == 1:
        axes = axes[None, :]
    for r, n in enumerate(cells):
        z = np.load(Path(a.data_dir) / f"{n}.npz"); x6 = z["x6"]; bbox = z["bbox"]
        _, H, W = x6.shape
        ndvi = load_ndvi_full(n, H, W) if a.multitemporal else None
        prob = cropland_prob(model, x6, dev, ndvi=ndvi)
        rgb = x6[0:3].transpose(1, 2, 0)

        g = load_county(a.dltb, n.split("_")[0], cache)
        tr = from_bounds(*[float(v) for v in bbox], W, H)
        idx = list(g.sindex.intersection(tuple(float(v) for v in bbox)))
        sub = g.iloc[idx].reset_index(drop=True)
        cb = shp_box(*bbox)
        keep = [j for j in range(len(sub)) if sub.geometry.iloc[j].intersects(cb)]
        sub = sub.iloc[keep].reset_index(drop=True)
        pid = rasterize([(sub.geometry.iloc[j], j + 1) for j in range(len(sub))],
                        out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        pred_map = np.zeros((H, W), np.uint8); true_map = np.zeros((H, W), np.uint8)
        pred_crop = []
        for j in range(len(sub)):
            m = pid == (j + 1)
            if not m.any():
                pred_crop.append(None); continue
            pc = 1 if prob[m].mean() >= 0.5 else 2
            tc = 1 if int(sub["cid"].iloc[j]) in (1, 2) else 2
            pred_map[m] = pc; true_map[m] = tc; pred_crop.append(pc == 1)
        sub["pred_cropland"] = [bool(c) if c is not None else None for c in pred_crop]
        sub["true_cropland"] = [int(c) in (1, 2) for c in sub["cid"]]
        sub[["DLBM", "cid", "pred_cropland", "true_cropland", "geometry"]].to_file(out / f"demo_{n}.geojson", driver="GeoJSON")

        ds = max(1, H // 1000)  # downsample for plotting
        axes[r, 0].imshow(rgb[::ds, ::ds]); axes[r, 0].set_title(f"{n}  1m RGB"); axes[r, 0].axis("off")
        axes[r, 1].imshow(pred_map[::ds, ::ds], cmap=CMAP, vmin=0, vmax=2)
        axes[r, 1].set_title("Predicted parcels (green=cropland)"); axes[r, 1].axis("off")
        axes[r, 2].imshow(true_map[::ds, ::ds], cmap=CMAP, vmin=0, vmax=2)
        axes[r, 2].set_title("DLTB ground-truth parcels"); axes[r, 2].axis("off")
    fig.tight_layout(); fig.savefig(out / "demo_parcels.png", dpi=160); plt.close(fig)
    print(f"[demo] saved demo_parcels.png + {len(cells)} GeoJSON to {out}", flush=True)


if __name__ == "__main__":
    main()
