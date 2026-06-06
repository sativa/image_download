"""Cross-province (Changzhi, Shanxi) parcel-level PRODUCT demo — visual proof of domain transfer.

Same as make_demo.py but Changzhi: a 6ch RGB model (zero adaptation) on Changzhi 1m imagery, parcels =
长治市 DLTB polygons. Renders [RGB | predicted cropland parcels | Changzhi DLTB ground truth] for a
few cells so the ~0.92 cross-province area-F1 is tangible. Also writes classified-parcel GeoJSON.
"""
import argparse, sys, time
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
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from changzhi_parcel_eval import prob1m

CMAP = ListedColormap(["#ffffff", "#3a9d23", "#d9c08a"])  # 0 nodata / 1 cropland / 2 other


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m_v2_bnd/best.pt")
    p.add_argument("--wrapper", action="store_true", default=True, help="DinoUNetBoundary (v2) ckpt")
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--dltb", default="/mnt/sda/zf/landform/data/changzhi_DLTB_wgs84.parquet")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/figures")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = a.device; t0 = time.time()

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    base = DinoUNet5ch(dino, num_classes=3, in_channels=6, unfreeze_last_n=4)
    if a.wrapper:
        from train_dino_1m_v2 import DinoUNetBoundary
        model = DinoUNetBoundary(base)
    else:
        model = base
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()
                           and model.state_dict()[k].shape == v.shape})
    model = model.to(dev).eval()

    g = gpd.read_parquet(a.dltb)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except Exception:
        g["geometry"] = g.geometry.buffer(0)
    cid = g["DLBM"].astype(str).str[:2]
    g["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0").astype(int)
    g = g[(g["cid"] >= 1) & (g["cid"] <= 12)].reset_index(drop=True); _ = g.sindex
    print(f"[demo-cz] {len(g)} polygons ({time.time()-t0:.0f}s)", flush=True)

    names = sorted(pth.stem for pth in Path(a.cz_1m_dir).glob("*.npz"))
    cells = names[:: max(1, len(names) // a.n)][:a.n]
    fig, axes = plt.subplots(len(cells), 3, figsize=(13, 4.3 * len(cells)))
    if len(cells) == 1:
        axes = axes[None, :]
    for r, n in enumerate(cells):
        z = np.load(Path(a.cz_1m_dir) / f"{n}.npz"); x6 = z["x6"]; bbox = z["bbox"]
        _, H, W = x6.shape
        prob = prob1m(model, x6, dev); rgb = x6[0:3].transpose(1, 2, 0)
        tr = from_bounds(*[float(v) for v in bbox], W, H)
        idx = list(g.sindex.intersection(tuple(float(v) for v in bbox)))
        cb = shp_box(*bbox); sub = g.iloc[idx].reset_index(drop=True)
        sub = sub.iloc[[j for j in range(len(sub)) if sub.geometry.iloc[j].intersects(cb)]].reset_index(drop=True)
        pid = rasterize([(sub.geometry.iloc[j], j + 1) for j in range(len(sub))],
                        out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        pred_map = np.zeros((H, W), np.uint8); true_map = np.zeros((H, W), np.uint8); pc_list = []
        for j in range(len(sub)):
            m = pid == (j + 1)
            if not m.any():
                pc_list.append(None); continue
            pc = 1 if prob[m].mean() >= 0.5 else 2
            tc = 1 if int(sub["cid"].iloc[j]) in (1, 2) else 2
            pred_map[m] = pc; true_map[m] = tc; pc_list.append(pc == 1)
        sub["pred_cropland"] = [bool(c) if c is not None else None for c in pc_list]
        sub["true_cropland"] = [int(c) in (1, 2) for c in sub["cid"]]
        sub[["DLBM", "DLMC", "cid", "pred_cropland", "true_cropland", "geometry"]].to_file(
            out / f"demo_changzhi_{n}.geojson", driver="GeoJSON")
        ds = max(1, H // 1000)
        axes[r, 0].imshow(rgb[::ds, ::ds]); axes[r, 0].set_title(f"Changzhi {n}  1m RGB"); axes[r, 0].axis("off")
        axes[r, 1].imshow(pred_map[::ds, ::ds], cmap=CMAP, vmin=0, vmax=2)
        axes[r, 1].set_title("Predicted (cross-province, green=cropland)"); axes[r, 1].axis("off")
        axes[r, 2].imshow(true_map[::ds, ::ds], cmap=CMAP, vmin=0, vmax=2)
        axes[r, 2].set_title("Changzhi DLTB ground truth"); axes[r, 2].axis("off")
        print(f"  cell {n} done ({time.time()-t0:.0f}s)", flush=True)
    fig.tight_layout(); fig.savefig(out / "demo_changzhi_parcels.png", dpi=160); plt.close(fig)
    print(f"[demo-cz] saved demo_changzhi_parcels.png + {len(cells)} GeoJSON", flush=True)


if __name__ == "__main__":
    main()
