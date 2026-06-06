"""Default classification backend: the trained DINOv3-Sat + FreqFusion + GDLX
binary cropland model (cropland = farmland + orchard).

Two guarantees the SAM-3 pipeline did not give:
  * FULL COVERAGE — every pixel is assigned cropland (1) or non-cropland (2)
    by thresholding the softmax cropland probability; there is no
    "unclassified" / class-0 background, so the polygon layer tiles the
    ENTIRE image with no gaps, even where the model is uncertain.
  * One trained model, not hand-tuned colour rules.

Single-source RGB input is duplicated into the model's two source slots
(Esri/Google are near-identical) and the 5 NDVI bands are zero-filled — the
model was trained with zero-NDVI cells, so this is in-distribution.
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


def run_cropland(cfg, device: str) -> None:
    """End-to-end binary cropland classification with full-coverage polygons."""
    import torch
    import torch.nn.functional as F  # noqa: F401 (used in _tiled_prob)
    import rasterio
    from .infer import read_rgb_from_geotiff, write_label_vector

    sd_dir = _sidecar_dir()
    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))
    from train_dino_1m_v3 import DinoV3FreqUNet
    from transformers import AutoModel

    _emit({"type": "stage", "stage": "reading_image"})
    rgb, profile, wgs84_bbox = read_rgb_from_geotiff(cfg.input_tif)
    H, W, _ = rgb.shape

    _emit({"type": "stage", "stage": "loading_model", "device": device, "backend": "cropland"})
    backbone = Path(cfg.backbone_dir)
    d3 = AutoModel.from_pretrained(str(backbone), local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=3, in_channels=11, unfreeze_last_n=4).to(device)
    sd = torch.load(cfg.weights, map_location=device, weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape},
                          strict=False)
    model.eval()

    rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8)
    x6 = np.concatenate([rgb_chw, rgb_chw], 0)                  # duplicate single source -> 6ch
    ndvi = np.zeros((5, H, W), np.float32)                      # zero NDVI (in-distribution)

    _emit({"type": "stage", "stage": "encoding_image", "height": H, "width": W, "backend": "cropland"})
    prob = _tiled_prob(model, x6, ndvi, device, cs=448)

    thr = cfg.confidence_threshold or 0.5
    # FULL COVERAGE: 1 = cropland, 2 = non-cropland; never 0. The whole image is tiled.
    label = np.where(prob >= thr, 1, 2).astype(np.uint8)

    _emit({"type": "stage", "stage": "polygonizing"})
    out_dir = cfg.output_tif.parent
    base_stem = cfg.output_tif.stem
    gpkg_path = out_dir / f"{base_stem}.gpkg"
    geojson_path = out_dir / f"{base_stem}.geojson"
    classes_meta = [(1, "cropland", (60, 180, 75)), (2, "non-cropland", (180, 180, 180))]
    write_label_vector(label, profile, classes_meta, gpkg_path, geojson_path)

    # Build a clean 1-band profile from scratch — never inherit the input's YCbCr/JPEG/tiled
    # creation options (they error on a 1-band deflate raster, e.g. BLOCKXSIZE without TILED).
    p = {"driver": "GTiff", "dtype": "uint8", "count": 1,
         "height": int(label.shape[0]), "width": int(label.shape[1]),
         "crs": profile["crs"], "transform": profile["transform"], "compress": "deflate"}
    with rasterio.open(cfg.output_tif, "w", **p) as dst:
        dst.write(label, 1)

    total = int(label.size)
    uniq, cnt = np.unique(label, return_counts=True)
    stats = {str(int(u)): {"pixels": int(c), "area_pct": round(100.0 * c / total, 3)}
             for u, c in zip(uniq.tolist(), cnt.tolist())}
    legend = {
        "classes": [{"id": 1, "label": "cropland", "rgb": [60, 180, 75]},
                    {"id": 2, "label": "non-cropland", "rgb": [180, 180, 180]}],
        "model": "DINOv3-Sat + FreqFusion + GDLX (binary cropland, full coverage)",
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
        "stats": stats,
    })


def _tiled_prob(model, x6, ndvi, dev: str, cs: int = 448) -> np.ndarray:
    """Tiled softmax cropland probability over the full image (overlap-averaged)."""
    import torch
    import torch.nn.functional as F
    from train_dino_1m import norm6

    _, SZ, SZw = x6.shape
    pad_h = max(0, cs - SZ)
    pad_w = max(0, cs - SZw)
    if pad_h or pad_w:                                          # pad small images up to one tile
        x6 = np.pad(x6, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        ndvi = np.pad(ndvi, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    _, PH, PW = x6.shape
    acc = np.zeros((PH, PW), np.float32)
    cnt = np.zeros((PH, PW), np.float32)
    ys = list(range(0, max(1, PH - cs + 1), cs))
    xs = list(range(0, max(1, PW - cs + 1), cs))
    if ys[-1] != PH - cs:
        ys.append(PH - cs)
    if xs[-1] != PW - cs:
        xs.append(PW - cs)
    use_amp = dev.startswith("cuda")
    total = len(ys) * len(xs)
    done = 0
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
                pr = torch.softmax(lg.float(), 1)[0, 1].cpu().numpy()
            acc[t:t + cs, l:l + cs] += pr
            cnt[t:t + cs, l:l + cs] += 1
            done += 1
            _emit({"type": "progress", "done": done, "total": total, "stage": "segmenting"})
    prob = acc / np.maximum(cnt, 1)
    return prob[:SZ, :SZw]                                      # crop padding back off
