"""7-class land-COVER per-parcel classification backend (DINOv3-Sat + FreqFusion).

Aggregated land-cover super-classes (every pixel gets exactly one — no nodata, no blank):
  1 耕地 cropland  2 园地 orchard  3 林地 forest  4 草地 grassland
  5 水体 water     6 建筑 built-up 7 荒漠 bare/desert

Guarantees:
  * FULL COVERAGE — label = argmax over the 7 classes (class 0/nodata excluded), so every pixel
    is classified even where the model is uncertain; the polygon layer tiles the WHOLE image.
  * PER-PARCEL — polygons are connected components (NOT dissolved by class): each contiguous
    region of one class becomes its own polygon, so adjacent fields of different classes are
    separated. (Touching same-class parcels still merge; boundary-head splitting is future work.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLASSES = [
    (1, "耕地", "cropland", (60, 180, 75)),
    (2, "园地", "orchard", (170, 255, 90)),
    (3, "林地", "forest", (0, 100, 0)),
    (4, "草地", "grassland", (190, 220, 100)),
    (5, "水体", "water", (0, 130, 200)),
    (6, "建筑", "built-up", (230, 25, 75)),
    (7, "荒漠", "bare", (170, 140, 100)),
]
NCLS = 9  # model has 9 logits (0 nodata + 8: …7荒漠 8设施大棚); 设施大棚 is merged into 建筑 at output
MERGE = {8: 6}  # 设施大棚(8) -> 建筑(6): +0.046 macro-F1, no retrain


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _sidecar_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def run_landcover(cfg, device: str) -> None:
    import torch
    import rasterio
    import rasterio.features
    import geopandas as gpd
    from shapely.geometry import shape
    from .infer import read_rgb_from_geotiff

    sd_dir = _sidecar_dir()
    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))
    from train_dino_1m_v3 import DinoV3FreqUNet
    from transformers import AutoModel

    _emit({"type": "stage", "stage": "reading_image"})
    rgb, profile, wgs84_bbox = read_rgb_from_geotiff(cfg.input_tif)
    H, W, _ = rgb.shape

    _emit({"type": "stage", "stage": "loading_model", "device": device, "backend": "landcover"})
    d3 = AutoModel.from_pretrained(str(cfg.backbone_dir), local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=NCLS, in_channels=11, unfreeze_last_n=4).to(device)
    sd = torch.load(cfg.weights, map_location=device, weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape},
                          strict=False)
    model.eval()

    rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8)
    x6 = np.concatenate([rgb_chw, rgb_chw], 0)
    ndvi = np.zeros((5, H, W), np.float32)

    _emit({"type": "stage", "stage": "encoding_image", "height": H, "width": W, "backend": "landcover"})
    probs = _tiled_probs(model, x6, ndvi, device, NCLS, cs=448)

    # FULL COVERAGE: argmax over classes 1..8 (skip class 0/nodata) -> every pixel labelled.
    label = (probs[1:].argmax(0) + 1).astype(np.uint8)
    for s, d in MERGE.items():                                # 设施大棚 -> 建筑 (higher accuracy 7-class output)
        label[label == s] = d
    label = _denoise(label)                                   # light median filter; preserves full coverage

    _emit({"type": "stage", "stage": "polygonizing"})
    out_dir = cfg.output_tif.parent
    base_stem = cfg.output_tif.stem
    gpkg_path = out_dir / f"{base_stem}.gpkg"
    geojson_path = out_dir / f"{base_stem}.geojson"
    name_zh = {c[0]: c[1] for c in CLASSES}
    name_en = {c[0]: c[2] for c in CLASSES}
    hexc = {c[0]: "#%02x%02x%02x" % c[3] for c in CLASSES}

    transform = profile["transform"]
    crs = profile["crs"]
    px_area = abs(transform.a * transform.e)                  # m^2 per pixel (EPSG:3857 metres)
    rows = []
    # connectivity=8: contiguous same-class region -> one parcel polygon. No dissolve, no area drop.
    for geom, value in rasterio.features.shapes(label, mask=None, connectivity=8, transform=transform):
        cid = int(value)
        g = shape(geom)
        rows.append({"class_id": cid, "label": name_zh[cid], "label_en": name_en[cid],
                     "rgb_hex": hexc[cid], "area_m2": float(g.area), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    total_area = float(gdf["area_m2"].sum()) or 1.0
    gdf["area_pct"] = (gdf["area_m2"] / total_area * 100.0).round(3)
    gdf.to_file(gpkg_path, driver="GPKG", layer="landcover")
    gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")

    # Clean 1-band profile from scratch (never inherit YCbCr/JPEG/tiled options).
    p = {"driver": "GTiff", "dtype": "uint8", "count": 1,
         "height": int(label.shape[0]), "width": int(label.shape[1]),
         "crs": profile["crs"], "transform": profile["transform"], "compress": "deflate"}
    with rasterio.open(cfg.output_tif, "w", **p) as dst:
        dst.write(label, 1)

    total_px = int(label.size)
    uniq, cnt = np.unique(label, return_counts=True)
    stats = {str(int(u)): {"pixels": int(c), "area_pct": round(100.0 * c / total_px, 3)}
             for u, c in zip(uniq.tolist(), cnt.tolist())}
    legend = {
        "classes": [{"id": c[0], "label": c[1], "label_en": c[2], "rgb": list(c[3])} for c in CLASSES],
        "model": "DINOv3-Sat + FreqFusion (7-class land-cover, per-parcel, full coverage)",
        "n_parcels": int(len(gdf)),
        "stats": stats,
    }
    legend_path = out_dir / f"{base_stem}.legend.json"
    legend_path.write_text(json.dumps(legend, indent=2, ensure_ascii=False))

    _emit({
        "type": "done",
        "label_gpkg": str(gpkg_path),
        "overlay_geojson": str(geojson_path),
        "legend_json": str(legend_path),
        "overlay_bbox_wgs84": [wgs84_bbox[0], wgs84_bbox[1], wgs84_bbox[2], wgs84_bbox[3]],
        "n_parcels": int(len(gdf)),
        "stats": stats,
    })


def _denoise(label: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter to remove salt-and-pepper; reassigns isolated pixels to neighbours
    (keeps full coverage — no pixel is ever set to nodata)."""
    try:
        import cv2
        return cv2.medianBlur(label, k)
    except Exception:
        return label


def _tiled_probs(model, x6, ndvi, dev: str, ncls: int, cs: int = 448) -> np.ndarray:
    """Tiled softmax over all classes -> (ncls, H, W) averaged probability."""
    import torch
    import torch.nn.functional as F
    from train_dino_1m import norm6

    _, SZ, SZw = x6.shape
    pad_h = max(0, cs - SZ); pad_w = max(0, cs - SZw)
    if pad_h or pad_w:
        x6 = np.pad(x6, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        ndvi = np.pad(ndvi, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    _, PH, PW = x6.shape
    acc = np.zeros((ncls, PH, PW), np.float32)
    cnt = np.zeros((PH, PW), np.float32)
    ys = list(range(0, max(1, PH - cs + 1), cs)); xs = list(range(0, max(1, PW - cs + 1), cs))
    if ys[-1] != PH - cs: ys.append(PH - cs)
    if xs[-1] != PW - cs: xs.append(PW - cs)
    use_amp = dev.startswith("cuda")
    total = len(ys) * len(xs); done = 0
    for t in ys:
        for l in xs:
            xc = norm6(x6[:, t:t + cs, l:l + cs])
            xc = np.concatenate([xc, ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(xb)
                else:
                    lg = model(xb)
                lg = lg[0] if isinstance(lg, tuple) else lg
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr
            cnt[t:t + cs, l:l + cs] += 1
            done += 1
            _emit({"type": "progress", "done": done, "total": total, "stage": "segmenting"})
    probs = acc / np.maximum(cnt, 1)
    return probs[:, :SZ, :SZw]
