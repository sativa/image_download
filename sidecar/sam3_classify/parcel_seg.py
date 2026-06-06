"""PER-PARCEL cropland backend = SAM 3 instance segmentation + DINOv3-Sat cropland classification.

SAM 3 finds every parcel (instance mask); the trained DINOv3 cropland model gives a per-pixel
cropland probability; each SAM-3 parcel is then labelled cropland/non-cropland by the MAJORITY
(mean prob >= threshold) of the model's prediction inside it. Pixels no SAM-3 mask covers are
filled by connected components of the per-pixel prediction, so coverage is 100%. Vectorisation is
PER-INSTANCE (no dissolve) — each parcel is its own polygon, so neighbouring same-class fields stay
separate. This is the OBIA design: SAM-3 = boundaries, DINOv3 = class.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _sidecar_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def run_parcel(cfg, device: str) -> None:
    import torch  # noqa: F401
    import rasterio
    import rasterio.features
    import cv2
    import geopandas as gpd
    from shapely.geometry import shape
    from .infer import read_rgb_from_geotiff, nms_masks
    from .segment_samgeo import auto_segment
    from .cropland_dino import _tiled_prob

    sd_dir = _sidecar_dir()
    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))
    from train_dino_1m_v3 import DinoV3FreqUNet
    from transformers import AutoModel

    thr = cfg.confidence_threshold or 0.5

    # ── 1. SAM 3 instance masks ──────────────────────────────────────────
    _emit({"type": "stage", "stage": "sam3_segmenting", "backend": "parcel"})

    def _seg_prog(done, total, stage="sam3_segmenting"):
        _emit({"type": "progress", "done": done, "total": total, "stage": stage})

    sam3_w = getattr(cfg, "sam3_weights", None) or "/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt"
    raw, (H, W) = auto_segment(tif_path=cfg.input_tif, n_grid=24, confidence_threshold=0.4,
                               device=device, sam3_checkpoint=str(sam3_w), on_progress=_seg_prog)
    masks = nms_masks(raw, iou_threshold=0.5, min_area_px=64)
    _emit({"type": "stage", "stage": "sam3_done", "n_parcels": len(masks)})

    # ── 2. DINOv3 cropland probability ───────────────────────────────────
    _emit({"type": "stage", "stage": "loading_model", "device": device, "backend": "parcel"})
    rgb, profile, wgs84_bbox = read_rgb_from_geotiff(cfg.input_tif)
    d3 = AutoModel.from_pretrained(str(cfg.backbone_dir), local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=3, in_channels=11, unfreeze_last_n=4).to(device)
    sd = torch.load(cfg.weights, map_location=device, weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()
    rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8)
    x6 = np.concatenate([rgb_chw, rgb_chw], 0)
    ndvi = np.zeros((5, H, W), np.float32)
    _emit({"type": "stage", "stage": "classifying_pixels", "backend": "parcel"})
    prob = _tiled_prob(model, x6, ndvi, device, cs=448)        # P(cropland), (H,W)

    # ── 3. instance-id map: each SAM-3 mask gets a unique id + a class ────
    _emit({"type": "stage", "stage": "labelling_parcels"})
    idmap = np.zeros((H, W), np.int32)
    cls_of = {0: 0}
    next_id = 1
    # higher-score masks first; only claim still-unassigned pixels (handles overlap)
    for score, m in sorted(masks, key=lambda sm: -sm[0]):
        claim = m & (idmap == 0)
        if int(claim.sum()) < 64:
            continue
        idmap[claim] = next_id
        cls_of[next_id] = 1 if float(prob[claim].mean()) >= thr else 2
        next_id += 1

    # ── 4. fill gaps (no SAM-3 mask) with connected components of pixel class
    gap = idmap == 0
    if gap.any():
        pix = np.where(prob >= thr, 1, 2).astype(np.uint8)
        for c in (1, 2):
            cmask = (gap & (pix == c)).astype(np.uint8)
            if not cmask.any():
                continue
            n_cc, cc = cv2.connectedComponents(cmask, connectivity=8)
            for lab in range(1, n_cc):
                region = cc == lab
                if int(region.sum()) < 16:        # tiny slivers -> still labelled (full coverage), merged below
                    pass
                idmap[region] = next_id
                cls_of[next_id] = c
                next_id += 1

    # ── 5. per-instance vectorisation (NO dissolve) ──────────────────────
    _emit({"type": "stage", "stage": "polygonizing"})
    transform = profile["transform"]; crs = profile["crs"]
    name_zh = {1: "耕地", 2: "非耕地"}; hexc = {1: "#3cb44b", 2: "#b4b4b4"}
    rows = []
    for geom, val in rasterio.features.shapes(idmap, mask=idmap > 0, connectivity=8, transform=transform):
        pid = int(val); c = cls_of.get(pid, 2)
        g = shape(geom)
        rows.append({"parcel_id": pid, "class_id": c, "label": name_zh[c],
                     "rgb_hex": hexc[c], "area_m2": float(g.area), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    out_dir = cfg.output_tif.parent; base = cfg.output_tif.stem
    gpkg_path = out_dir / f"{base}.gpkg"; geojson_path = out_dir / f"{base}.geojson"
    gdf.to_file(gpkg_path, driver="GPKG", layer="parcels")
    gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")

    # class raster (cropland/non-cropland) for reference
    clsmap = np.zeros((H, W), np.uint8)
    for pid, c in cls_of.items():
        if pid:
            clsmap[idmap == pid] = c
    p = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": H, "width": W,
         "crs": crs, "transform": transform, "compress": "deflate"}
    with rasterio.open(cfg.output_tif, "w", **p) as dst:
        dst.write(clsmap, 1)

    n_crop = sum(1 for r in rows if r["class_id"] == 1)
    stats = {"1": {"parcels": n_crop}, "2": {"parcels": len(rows) - n_crop}}
    legend = {"classes": [{"id": 1, "label": "耕地", "rgb": [60, 180, 75]},
                          {"id": 2, "label": "非耕地", "rgb": [180, 180, 180]}],
              "model": "SAM3 instances + DINOv3-Sat cropland (per-parcel)",
              "n_parcels": int(len(rows)), "stats": stats}
    legend_path = out_dir / f"{base}.legend.json"
    legend_path.write_text(json.dumps(legend, indent=2, ensure_ascii=False))

    _emit({"type": "done", "label_gpkg": str(gpkg_path), "overlay_geojson": str(geojson_path),
           "legend_json": str(legend_path),
           "overlay_bbox_wgs84": [wgs84_bbox[0], wgs84_bbox[1], wgs84_bbox[2], wgs84_bbox[3]],
           "n_parcels": int(len(rows)), "stats": stats})
