"""GUI/CLI backend — BEST per-parcel delineation line: DINOv3-Sat distance head -> dist-peak watershed
(Hann-blended tiling, no seams) for cropland + classifier connected-components for other classes ->
full-coverage 7-class instances -> GeoParquet (default) + GeoJSON + GPKG + class raster.

This is the deployed form of the tuned dist-peak route (Gansu area-match 98%, Yuzhong county 90.6%,
cross-province 95%). Weights = the distance-head checkpoint (dino_v3_bdd -> ~/D/cropland_dino/parcel_dist.pt)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _sidecar_dir() -> Path:
    return Path(__file__).resolve().parent.parent


class _Params:
    """dist-peak watershed params (tuned optimum md20/pt0.4/ma200); GUI may override via cfg."""
    def __init__(self, cfg):
        self.min_dist = int(getattr(cfg, "min_dist", 0) or 20)
        self.peak_thr = float(getattr(cfg, "peak_thr", 0) or 0.4)
        self.min_area_px = int(getattr(cfg, "min_marker", 0) or 200)
        self.ridge = bool(getattr(cfg, "ridge", False))


def run_parcel_dist(cfg, device: str) -> None:
    import torch  # noqa: F401
    import rasterio
    import rasterio.features
    import geopandas as gpd
    from shapely.geometry import shape
    from .infer import read_rgb_from_geotiff

    sd = _sidecar_dir()
    if str(sd) not in sys.path:
        sys.path.insert(0, str(sd))
    from train_dino_1m_v3 import DinoV3FreqUNetBDD
    from transformers import AutoModel
    from dino_parcel_eval import infer_heads, dist_peak_instances  # noqa: F401 (dist_peak_instances via build_idmap)
    from dino_parcel_export import build_idmap, NAME_ZH, NAME_EN, HEX, RGB, CLASSES

    _emit({"type": "stage", "stage": "reading_image"})
    rgb, profile, wgs84_bbox = read_rgb_from_geotiff(cfg.input_tif)
    H, W, _ = rgb.shape

    _emit({"type": "stage", "stage": "loading_model", "device": device, "backend": "parcel_dist"})
    d3 = AutoModel.from_pretrained(str(cfg.backbone_dir), local_files_only=True)
    model = DinoV3FreqUNetBDD(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(device)
    sd_ = torch.load(cfg.weights, map_location=device, weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd_.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()

    rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8)
    x6 = np.concatenate([rgb_chw, rgb_chw], 0)                     # GeoTIFF is one source -> duplicate to 6-ch

    _emit({"type": "stage", "stage": "classifying_pixels", "backend": "parcel_dist"})
    dev = device if str(device).startswith("cuda") else "cpu"
    clsprob, dist, bnd = infer_heads(model, x6, dev)               # Hann-blended tiling (no seams)

    _emit({"type": "stage", "stage": "delineating_parcels"})
    idmap, cls_of = build_idmap(clsprob, dist, bnd, _Params(cfg))  # dist-peak cropland + CC others, full coverage

    _emit({"type": "stage", "stage": "polygonizing"})
    transform = profile["transform"]; crs = profile["crs"]
    pix_m = (float(wgs84_bbox[2]) - float(wgs84_bbox[0])) * 111320 * math.cos(
        math.radians((float(wgs84_bbox[1]) + float(wgs84_bbox[3])) / 2)) / W
    simp = float(getattr(cfg, "simplify_px", 0) or 2.0) * abs(transform.a)
    areas_px = np.bincount(idmap.ravel())
    rows = []
    for geom, val in rasterio.features.shapes(idmap, mask=idmap > 0, connectivity=8, transform=transform):
        pid = int(val); c = cls_of.get(pid)
        if not c:
            continue
        g = shape(geom)
        if simp > 0:
            gs = g.simplify(simp, preserve_topology=True)
            g = gs if (not gs.is_empty and gs.is_valid) else g
        area_m2 = float(areas_px[pid]) * pix_m * pix_m if pid < len(areas_px) else 0.0
        rows.append({"parcel_id": pid, "class_id": c, "label": NAME_ZH[c], "label_en": NAME_EN[c],
                     "rgb_hex": HEX[c], "area_m2": round(area_m2, 1), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    out_dir = cfg.output_tif.parent; base = cfg.output_tif.stem
    parquet_path = out_dir / f"{base}.parquet"; gpkg_path = out_dir / f"{base}.gpkg"
    geojson_path = out_dir / f"{base}.geojson"
    gdf_wgs = gdf.to_crs("EPSG:4326")
    gdf_wgs.to_parquet(parquet_path)                               # GeoParquet — default storage
    gdf.to_file(gpkg_path, driver="GPKG", layer="parcels")
    gdf_wgs.to_file(geojson_path, driver="GeoJSON")

    clsmap = np.zeros((H, W), np.uint8)
    for pid, c in cls_of.items():
        if pid:
            clsmap[idmap == pid] = c
    pr = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": H, "width": W,
          "crs": crs, "transform": transform, "compress": "deflate"}
    with rasterio.open(cfg.output_tif, "w", **pr) as dst:
        dst.write(clsmap, 1)

    from collections import Counter
    cc = Counter(r["class_id"] for r in rows)
    stats = {str(k): {"parcels": int(v)} for k, v in cc.items()}
    legend = {"classes": [{"id": c[0], "label": c[1], "label_en": c[2], "rgb": list(c[3])} for c in CLASSES],
              "model": "DINOv3-Sat distance head -> dist-peak watershed (Hann blend) + 7-class (per-parcel)",
              "n_parcels": int(len(rows)), "stats": stats}
    legend_path = out_dir / f"{base}.legend.json"
    legend_path.write_text(json.dumps(legend, indent=2, ensure_ascii=False))

    _emit({"type": "done", "label_gpkg": str(gpkg_path), "label_parquet": str(parquet_path),
           "overlay_geojson": str(geojson_path), "legend_json": str(legend_path),
           "overlay_bbox_wgs84": [wgs84_bbox[0], wgs84_bbox[1], wgs84_bbox[2], wgs84_bbox[3]],
           "n_parcels": int(len(rows)), "stats": stats})
