"""Layered per-parcel backend (no SAM3): the model's OWN parcel-boundary head delineates parcels,
then the classification head assigns each parcel a land-cover type. Mirrors the FSDA recipe
(extent + boundary -> refined parcels) but with one DINOv3-Sat model + two heads.

Layer 1 (划分边界): boundary-head probability -> watershed -> parcel instances (full coverage).
Layer 2 (赋类型):   per instance, argmax of the mean class-head probability -> 8-class land-cover.
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
NCLS = 9  # model has 9 logits (…7荒漠 8设施大棚); 设施大棚 merged into 建筑 at output (+0.046 macro-F1)
MERGE = {8: 6}  # 设施大棚(8) -> 建筑(6)
# FSDA-style boundary refinement (clean parcel topology, no micro-slivers):
MIN_MARKER_PX = 150   # drop watershed markers smaller than this -> micro-basins merge into neighbours
SIMPLIFY_PX = 2.0     # Douglas-Peucker simplify tolerance (in pixels) for smoother vector edges


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _sidecar_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def run_parcel_bh(cfg, device: str) -> None:
    import torch
    import rasterio
    import rasterio.features
    import geopandas as gpd
    from shapely.geometry import shape
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from .infer import read_rgb_from_geotiff

    sd_dir = _sidecar_dir()
    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))
    from train_dino_1m_v3 import DinoV3FreqUNetBD as DinoV3FreqUNet
    from transformers import AutoModel

    _emit({"type": "stage", "stage": "reading_image"})
    rgb, profile, wgs84_bbox = read_rgb_from_geotiff(cfg.input_tif)
    H, W, _ = rgb.shape

    _emit({"type": "stage", "stage": "loading_model", "device": device, "backend": "parcel_bh"})
    d3 = AutoModel.from_pretrained(str(cfg.backbone_dir), local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=NCLS, in_channels=11, unfreeze_last_n=4).to(device)
    sd = torch.load(cfg.weights, map_location=device, weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()

    rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8)
    x6 = np.concatenate([rgb_chw, rgb_chw], 0)
    ndvi = np.zeros((5, H, W), np.float32)

    _emit({"type": "stage", "stage": "classifying_pixels", "backend": "parcel_bh"})
    clsprob, bndprob = _tiled_cls_bnd(model, x6, ndvi, device, NCLS, cs=448)

    # ── Layer 1: delineate parcels by watershed on the boundary probability ──
    _emit({"type": "stage", "stage": "delineating_parcels"})
    min_marker = int(getattr(cfg, "min_marker", 0) or MIN_MARKER_PX)   # GUI-tunable granularity
    simp_px = float(getattr(cfg, "simplify_px", 0) or SIMPLIFY_PX)
    marker_thr = 0.30
    # FSDA-style boundary connection: hysteresis (extend weak edges connected to strong) + morphological
    # closing (bridge small gaps) -> close broken boundaries so under-segmented parcels separate.
    try:
        import cv2
        from skimage.filters import apply_hysteresis_threshold
        edge = apply_hysteresis_threshold(bndprob, 0.22, 0.5).astype(np.uint8)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ridge = np.maximum(bndprob, edge.astype(np.float32))   # connected ridge for watershed elevation
    except Exception:
        ridge = bndprob
    seeds = (ridge < marker_thr)                               # interior (low-boundary) pixels
    markers, _ = ndi.label(seeds)                              # one marker per parcel interior
    if min_marker > 0:                                          # drop micro-markers -> tiny basins merge into neighbour
        cnt = np.bincount(markers.ravel())
        small = np.nonzero(cnt < min_marker)[0]; small = small[small != 0]
        if small.size:
            markers[np.isin(markers, small)] = 0
    inst = watershed(ridge, markers)                           # flood to (connected) boundaries -> full coverage

    # ── Layer 2: classify each parcel by mean class probability ──────────────
    _emit({"type": "stage", "stage": "labelling_parcels"})
    pix_cls = clsprob[1:].argmax(0) + 1                        # per-pixel fallback (1..8)
    for s, d in MERGE.items():
        pix_cls[pix_cls == s] = d                              # 设施大棚 -> 建筑
    n_inst = int(inst.max())
    cls_of = {0: 0}
    flat = inst.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_inst = flat[order]
    bounds = np.searchsorted(sorted_inst, np.arange(1, n_inst + 2))
    clsflat = clsprob.reshape(NCLS, -1)
    for pid in range(1, n_inst + 1):
        sel = order[bounds[pid - 1]:bounds[pid]]
        if sel.size == 0:
            continue
        mean_p = clsflat[1:, sel].mean(axis=1)                 # mean class prob over the parcel
        c = int(mean_p.argmax()) + 1
        cls_of[pid] = MERGE.get(c, c)                          # 设施大棚 -> 建筑

    _emit({"type": "stage", "stage": "polygonizing"})
    transform = profile["transform"]; crs = profile["crs"]
    name_zh = {c[0]: c[1] for c in CLASSES}; name_en = {c[0]: c[2] for c in CLASSES}
    hexc = {c[0]: "#%02x%02x%02x" % c[3] for c in CLASSES}
    simp = simp_px * abs(transform.a)                           # Douglas-Peucker tolerance in CRS units
    rows = []
    for geom, val in rasterio.features.shapes(inst.astype(np.int32), mask=inst > 0,
                                              connectivity=4, transform=transform):
        pid = int(val); c = cls_of.get(pid, int(pix_cls.flat[0]))
        if c < 1:
            continue
        g = shape(geom)
        if simp > 0:
            g = g.simplify(simp, preserve_topology=True)        # smoother field edges (Douglas-Peucker)
            if g.is_empty or not g.is_valid:
                g = shape(geom)
        rows.append({"parcel_id": pid, "class_id": c, "label": name_zh[c], "label_en": name_en[c],
                     "rgb_hex": hexc[c], "area_m2": float(g.area), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    out_dir = cfg.output_tif.parent; base = cfg.output_tif.stem
    gpkg_path = out_dir / f"{base}.gpkg"; geojson_path = out_dir / f"{base}.geojson"
    gdf.to_file(gpkg_path, driver="GPKG", layer="parcels")
    gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")

    clsmap = np.zeros((H, W), np.uint8)
    for pid, c in cls_of.items():
        if pid:
            clsmap[inst == pid] = c
    p = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": H, "width": W,
         "crs": crs, "transform": transform, "compress": "deflate"}
    with rasterio.open(cfg.output_tif, "w", **p) as dst:
        dst.write(clsmap, 1)

    from collections import Counter
    cc = Counter(r["class_id"] for r in rows)
    stats = {str(k): {"parcels": int(v)} for k, v in cc.items()}
    legend = {"classes": [{"id": c[0], "label": c[1], "label_en": c[2], "rgb": list(c[3])} for c in CLASSES],
              "model": "DINOv3-Sat boundary head (watershed) + 8-class head (per-parcel, layered)",
              "n_parcels": int(len(rows)), "stats": stats}
    legend_path = out_dir / f"{base}.legend.json"
    legend_path.write_text(json.dumps(legend, indent=2, ensure_ascii=False))

    _emit({"type": "done", "label_gpkg": str(gpkg_path), "overlay_geojson": str(geojson_path),
           "legend_json": str(legend_path),
           "overlay_bbox_wgs84": [wgs84_bbox[0], wgs84_bbox[1], wgs84_bbox[2], wgs84_bbox[3]],
           "n_parcels": int(len(rows)), "stats": stats})


def _tiled_cls_bnd(model, x6, ndvi, dev: str, ncls: int, cs: int = 448):
    """Tiled inference -> (class softmax (ncls,H,W), boundary sigmoid (H,W))."""
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
    accb = np.zeros((PH, PW), np.float32)
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
                        cls_lg, bnd_lg, _ = model(xb)
                else:
                    cls_lg, bnd_lg, _ = model(xb)
                if cls_lg.shape[-2:] != (cs, cs):
                    cls_lg = F.interpolate(cls_lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    bnd_lg = F.interpolate(bnd_lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(cls_lg.float(), 1)[0].cpu().numpy()
                pb = torch.sigmoid(bnd_lg.float())[0, 0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr
            accb[t:t + cs, l:l + cs] += pb
            cnt[t:t + cs, l:l + cs] += 1
            done += 1
            _emit({"type": "progress", "done": done, "total": total, "stage": "classifying_pixels"})
    cnt = np.maximum(cnt, 1)
    return (acc / cnt)[:, :SZ, :SZw], (accb / cnt)[:SZ, :SZw]
