"""Re-evaluate v18 best.pt with REALIGNED labels.

Bug: original code rasterized polygons using metadata bbox + TIF transform.
Most TIFs snap to z17 tile boundaries (~200m offset) or were reused from
older selections (~24km offset). Fix: use the TIF's actual bounds when
clipping polygons.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DEFAULT_DINOV2, IMAGENET_MEAN, IMAGENET_STD, DLBM_TO_CLASS, DinoUNet
from train_v17_cropland import evaluate_full_binary_tta


def rasterise_dltb_binary_realigned(gdf_wgs84, tif_bounds, tif_crs, transform, H, W):
    """Rasterise using the TIF's ACTUAL bounds, not the metadata bbox.

    gdf_wgs84: polygons in EPSG:4326
    tif_bounds: rasterio BoundingBox in tif_crs (e.g., EPSG:3857)
    """
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from pyproj import Transformer
    # Convert TIF bounds to WGS84 for sindex query
    t = Transformer.from_crs(tif_crs, "EPSG:4326", always_xy=True)
    left, bottom = t.transform(tif_bounds.left, tif_bounds.bottom)
    right, top = t.transform(tif_bounds.right, tif_bounds.top)
    bb_wgs84 = (left, bottom, right, top)

    idx = list(gdf_wgs84.sindex.intersection(bb_wgs84))
    sub = gdf_wgs84.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bb_wgs84))
    sub = sub[~sub.geometry.is_empty]
    if len(sub) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    sub_proj = sub.to_crs(tif_crs)
    sub_proj["bin"] = np.where((sub["cid"] == 1) | (sub["cid"] == 2), 1, 2)
    shapes = [(g, int(c)) for g, c in zip(sub_proj.geometry, sub_proj["bin"])]
    return rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--checkpoint", type=Path, default=HOME / "results/v18/best.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use-tta", action="store_true")
    args = p.parse_args()

    regions_meta = json.loads(args.regions_json.read_text())
    test = regions_meta["test"]
    print(f"[1] {len(test)} test regions, TTA={args.use_tta}", flush=True)

    import geopandas as gpd, rasterio
    gdf_per_county = {}
    for r in test:
        code = r["county"]
        if code in gdf_per_county: continue
        g = gpd.read_parquet(args.dltb_cache / f"{code}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g

    test_regions_old = []
    test_regions_new = []
    for r in test:
        bb = tuple(r["bbox"])
        for src in ["esri", "google"]:
            path = args.data_cache / f"{r['county']}_{r['idx']}_{src}.tif"
            if not path.exists(): continue
            with rasterio.open(path) as rs:
                bands = rs.read(out_dtype="uint8")
                tif_bounds = rs.bounds
                tif_crs = rs.crs
                transform = rs.transform
                H, W = rs.height, rs.width
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            # OLD: use metadata bbox
            from train_v16_binary import rasterise_dltb_binary
            lbl_old = rasterise_dltb_binary(gdf_per_county[r["county"]], bb, transform, H, W)
            # NEW: use actual TIF bounds
            lbl_new = rasterise_dltb_binary_realigned(
                gdf_per_county[r["county"]], tif_bounds, tif_crs, transform, H, W)
            test_regions_old.append((rgb, lbl_old))
            test_regions_new.append((rgb, lbl_new))
            valid_old = (lbl_old > 0).sum()
            valid_new = (lbl_new > 0).sum()
            print(f"  {r['county']}_{r['idx']}_{src}: valid_old={valid_old:,} valid_new={valid_new:,} "
                  f"(+{(valid_new-valid_old)/max(valid_old,1)*100:.0f}%)", flush=True)

    print(f"\n[2] Load v18 best.pt", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=3, unfreeze_last_n=4).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model = getattr(model, "eval")()

    for name, regions in [("OLD (metadata bbox)", test_regions_old),
                           ("NEW (TIF actual bounds)", test_regions_new)]:
        print(f"\n[3] Eval with {name} labels", flush=True)
        t0 = time.time()
        ms = []
        for rgb, lbl in regions:
            m = evaluate_full_binary_tta(model, rgb, lbl, args.device,
                                          stride=384, batch_size=8,
                                          use_tta=args.use_tta)
            if m: ms.append(m)
        avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"  acc={avg['acc']:.3f} iou={avg['iou']:.3f} "
              f"prec={avg['precision']:.3f} rec={avg['recall']:.3f} F1={avg['f1']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
