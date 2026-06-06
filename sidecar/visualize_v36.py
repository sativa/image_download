"""Visualize v36 (SOTA, F1=0.865) predictions on 8 balanced test cells.
4-panel: z17 RGB | S2 NDVI | GT mask | v36 prediction overlay.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS
from train_v33_multitemporal import EXTRA_YEARS as Y4

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3


def stretch(arr, pct=2):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, pct), np.percentile(arr, 100 - pct)
    return np.clip((arr - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def colorize_ndvi(ndvi):
    H, W = ndvi.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    n = np.clip((ndvi + 0.2) / 1.0, 0, 1)
    out[..., 0] = (np.clip(1 - n * 2, 0, 1) * 255).astype(np.uint8)
    out[..., 1] = (np.clip(n, 0, 1) * 255).astype(np.uint8)
    out[..., 2] = (np.clip(0.3 - n, 0, 1) * 255).astype(np.uint8)
    return out


def overlay_pred_vs_gt(rgb, pred_crop, gt_crop, alpha=0.55):
    H, W = pred_crop.shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    pi = pred_crop > 0; ti = gt_crop > 0
    overlay[(pi & ti)] = [0, 255, 0, int(255 * alpha)]
    overlay[(pi & ~ti)] = [255, 0, 0, int(255 * alpha)]
    overlay[(~pi & ti)] = [255, 165, 0, int(255 * alpha)]
    base = Image.fromarray(rgb).convert("RGBA")
    return Image.alpha_composite(base, Image.fromarray(overlay))


def upsample_to(arr2d, H, W):
    t = torch.from_numpy(arr2d.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--ndvi-yr-dir", type=Path, default=HOME / "data/v33_ndvi_multitemporal")
    p.add_argument("--z17-dir", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--ckpt", type=Path, default=HOME / "results/v36/best.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v36_viz")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] load v36", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=args.backbone, encoder_weights=None,
                     in_channels=9, classes=3).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model = getattr(model, "eval")()

    print("[2] load test cells", flush=True)
    regions = json.loads(args.regions_json.read_text())
    import geopandas as gpd, rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from affine import Affine
    gdf = {}
    for r in regions["test"]:
        c = r["county"]
        if c in gdf: continue
        g = gpd.read_parquet(args.dltb_cache / f"{c}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326: g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf[c] = g

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    cjk_cands = ["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei",
                  "AR PL UMing CN", "SimHei"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((f for f in cjk_cands if f in avail), None)
    if cjk:
        plt.rcParams["font.family"] = [cjk, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    for r in regions["test"]:
        bb = tuple(r["bbox"])
        name = f"{r['county']}_{r['idx']}"
        s2_path = args.s2_dir / f"{name}.npz"
        nd_path = args.ndvi_yr_dir / f"{name}.npz"
        if not (s2_path.exists() and nd_path.exists()):
            print(f"  skip {name}: no data"); continue
        s2 = np.load(s2_path); nd = np.load(nd_path)
        rgbnir = s2["rgbnir"]; ndvi_s2 = s2["ndvi"]
        Hs, Ws = rgbnir.shape[1], rgbnir.shape[2]
        years = nd["years"].tolist()
        stack = nd["ndvi_years"].astype(np.float32) / 10000.0
        stack_sel = np.stack([stack[years.index(y)] for y in Y4 if y in years], 0)
        ndvi_years_up = np.stack([upsample_to(stack_sel[i], Hs, Ws)
                                   for i in range(len(stack_sel))], 0).astype(np.float32)
        # Build 9-ch input
        x = rgbnir.astype(np.float32).copy()
        for b in range(4): x[b] = (x[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_s2_n = (ndvi_s2.astype(np.float32) - NDVI_MEAN) / NDVI_STD
        ndvi_years_n = (ndvi_years_up - NDVI_MEAN) / NDVI_STD
        x9 = np.concatenate([x, ndvi_s2_n[None], ndvi_years_n], 0).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x9)[None].to(args.device)
            pred_s2 = model(xt).argmax(dim=1)[0].cpu().numpy()
        pred_crop_s2 = (pred_s2 == 1).astype(np.uint8)

        # z17 RGB for visualization
        rgb_z17 = None; H_z = W_z = None
        for src in ["esri", "google"]:
            p_z = args.z17_dir / f"{name}_{src}.tif"
            if p_z.exists():
                with rasterio.open(p_z) as rs:
                    bands = rs.read(out_dtype="uint8")
                    rgb_z17 = np.stack([bands[0], bands[1], bands[2]], axis=-1)
                    H_z, W_z = rs.height, rs.width
                    transform_z = rs.transform
                break
        if rgb_z17 is None:
            print(f"  skip {name}: no z17"); continue

        # GT at z17 grid via rasterize w/ to_crs to 3857 since z17 is 3857
        from train_v16_binary import rasterise_dltb_binary
        gt_z = rasterise_dltb_binary(gdf[r["county"]], bb, transform_z, H_z, W_z)
        gt_crop_z = (gt_z == 1).astype(np.uint8)

        # Upsample pred to z17 grid
        pred_t = torch.from_numpy(pred_crop_s2.astype(np.float32))[None, None]
        pred_up = F.interpolate(pred_t, size=(H_z, W_z), mode="nearest").numpy()[0, 0]
        pred_crop_z = (pred_up > 0.5).astype(np.uint8)

        valid_z = gt_z > 0
        f1 = 0
        if valid_z.any():
            tp = int(((pred_crop_z == 1) & (gt_crop_z == 1) & valid_z).sum())
            fp = int(((pred_crop_z == 1) & (gt_crop_z == 0) & valid_z).sum())
            fn = int(((pred_crop_z == 0) & (gt_crop_z == 1) & valid_z).sum())
            prec = tp/(tp+fp) if tp+fp else 0
            rec = tp/(tp+fn) if tp+fn else 0
            f1 = 2*prec*rec/(prec+rec) if prec+rec else 0

        ndvi_avg = np.nanmean(ndvi_s2)
        ndvi_yr_var = np.std([np.nanmean(ndvi_years_up[i]) for i in range(len(Y4))])

        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes[0,0].imshow(rgb_z17); axes[0,0].set_title("z17 RGB (~1m)", fontsize=12); axes[0,0].axis("off")
        ndvi_color = colorize_ndvi(ndvi_s2)
        axes[0,1].imshow(ndvi_color)
        axes[0,1].set_title(f"S2 NDVI 2021  mean={ndvi_avg:.2f}, 4yr-var={ndvi_yr_var:.3f}", fontsize=12)
        axes[0,1].axis("off")
        gt_overlay = np.zeros((H_z, W_z, 4), dtype=np.uint8)
        gt_overlay[gt_crop_z > 0] = [0, 200, 0, 140]
        gt_overlay[(gt_z == 2)] = [180, 100, 50, 140]
        gt_img = Image.alpha_composite(Image.fromarray(rgb_z17).convert("RGBA"),
                                         Image.fromarray(gt_overlay))
        axes[1,0].imshow(gt_img)
        axes[1,0].set_title(f"GT (绿=耕地, 棕=其他)", fontsize=12); axes[1,0].axis("off")
        pred_img = overlay_pred_vs_gt(rgb_z17, pred_crop_z, gt_crop_z)
        axes[1,1].imshow(pred_img)
        axes[1,1].set_title(f"v36 prediction (绿=TP, 红=FP, 橙=FN)  F1={f1:.3f}",
                              fontsize=12, color="darkblue", weight="bold")
        axes[1,1].axis("off")
        plt.suptitle(f"{name}  ({bb[0]:.3f}°E, {bb[1]:.3f}°N)", fontsize=14)
        plt.tight_layout()
        out = args.out_dir / f"{name}_F1{int(f1*1000):03d}.png"
        plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
        print(f"  {name}: F1={f1:.3f} → {out.name}", flush=True)
    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
