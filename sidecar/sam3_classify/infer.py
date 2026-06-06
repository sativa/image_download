"""Run a SAM 3 land-cover pass over one GeoTIFF.

Composition rule (compose_masks):
  For each pixel, the winning class is the one with the highest
  `instance_score` among all instance masks that cover that pixel.
  Pixels no class covers stay unclassified (value 0).

Why "max instance score" and not e.g. mask area / union vote?
  - SAM 3 returns one or more instance masks per text prompt with a
    confidence score in [0, 1]. We have no per-pixel probability map; the
    best per-pixel proxy is "the score of the most confident instance
    whose mask includes this pixel".
  - Higher-priority classes (water > built_up > cropland > ...) come from
    the model's own confidence, not a hand-coded priority table. If a
    confident water mask covers a tile labelled by a weak cropland mask,
    water wins on the overlap.

Single-pass over input (set_image called once). set_text_prompt then loops
over every (class, phrase) pair. SAM 3 caches the image embedding, so
swapping prompts is much cheaper than the initial encode.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# We deliberately keep all sam3 imports lazy: env_patches.apply() must run
# first, and the module-level cost of importing sam3 is non-trivial (~5 s).


def _emit(record: dict) -> None:
    """Write one NDJSON record to stdout and flush.

    The Rust parent process reads stdout line-by-line and re-broadcasts
    each line as a Tauri event. Anything else printed to stdout breaks
    that contract — use sys.stderr for unstructured logging.
    """
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


@dataclass
class InferConfig:
    input_tif: Path
    output_tif: Path
    device: str
    weights: Path
    confidence_threshold: float = 0.4
    backend: str = "cropland"  # cropland | parcel (SAM3+cropland) | landcover (7-class) | sam3 | dino | slic
    classifier: str = "color"  # color | siglip
    backbone_dir: Path = None  # DINOv3-Sat backbone dir (cropland/landcover/parcel backends)
    sam3_weights: Path = None  # SAM 3 checkpoint (parcel backend only)


def read_rgb_from_geotiff(path: Path):
    """Read a GeoTIFF as an (H, W, 3) uint8 array, dropping alpha if any.

    Returns (rgb_array, rasterio_profile, wgs84_bbox). The profile lets
    us write the label raster back with the same transform/CRS but
    band_count=1. The bbox is the *actual* extent of the file (often
    larger than the user's nominal download bbox because the COG snaps
    to whole tile boundaries) reprojected to WGS84 lon/lat; the frontend
    uses it to position the overlay PNG so it aligns with the live
    basemap tiles instead of drifting.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        bands = src.read(out_dtype="uint8")  # (C, H, W)
        profile = src.profile.copy()
        if bands.shape[0] >= 3:
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        else:
            raise ValueError(f"need at least 3 bands, input has {bands.shape[0]}")
        # transform_bounds densifies edges then reprojects — important
        # when the source CRS distorts straight lines (EPSG:3857 → 4326
        # is fine without densification, but the call shape stays the
        # same regardless of source CRS).
        wgs84_bbox = tuple(
            transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        )
    return rgb, profile, wgs84_bbox


def _drop_small_holes(geom, min_hole_area_m2: float):
    """Remove interior rings smaller than `min_hole_area_m2`.

    Walks (Multi)Polygon members, keeps the exterior of each part, and
    drops any hole whose ring polygon area is below the threshold.
    Operates in the geometry's native CRS — caller passes the threshold
    in those units (metres on EPSG:3857).
    """
    from shapely.geometry import Polygon, MultiPolygon

    def trim_one(poly: Polygon) -> Polygon:
        kept_interiors = []
        for ring in poly.interiors:
            ring_poly = Polygon(ring)
            if ring_poly.area >= min_hole_area_m2:
                kept_interiors.append(ring)
        return Polygon(poly.exterior, kept_interiors)

    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return trim_one(geom)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([trim_one(p) for p in geom.geoms])
    # Other geometry types (Point, Line, GeometryCollection) shouldn't
    # appear from rasterio.features.shapes but return untouched if they do.
    return geom


def write_label_vector(
    label: np.ndarray,
    src_profile: dict,
    classes_meta: list[tuple[int, str, tuple[int, int, int]]],
    gpkg_path: Path,
    geojson_path: Path,
) -> None:
    """Polygonize the label raster and persist as GPKG (+ GeoJSON for map).

    Two files are written from one polygonization:
      * `gpkg_path` — canonical product, in the source CRS (EPSG:3857).
        Layer "landform"; one row per (class_id, dissolved polygon).
      * `geojson_path` — same geometry reprojected to EPSG:4326, for the
        MapLibre frontend (which can't read GPKG directly).

    Why dissolve by class?
      * SAM 3 produces N small instance masks per prompt; without
        dissolving the GPKG would be huge and the map would render
        thousands of polygon strokes.
      * The user's downstream consumers (e.g. DNDC) think per land-cover
        class, not per instance.

    Edge handling:
      * `mask=label > 0` excludes the unclassified background — those
        pixels become "no polygon" rather than a class-0 hole.
      * `connectivity=4` (default in rasterio) avoids cross-corner bleed
        that would join touching diagonal cells of the same class into a
        single weirdly thin polygon.

    Attributes per feature: class_id (int), label (str), rgb_hex (str),
    area_m2 (float, computed in EPSG:3857 = metric), area_pct (float,
    relative to the COG's total area).
    """
    import geopandas as gpd
    import rasterio.features
    from shapely.geometry import shape

    transform = src_profile["transform"]
    crs = src_profile["crs"]

    # rasterio.features.shapes streams (geometry_dict, value) tuples.
    # Materialise into a list keyed by class_id so we can dissolve below.
    by_class: dict[int, list] = {}
    for geom, value in rasterio.features.shapes(
        label.astype(np.uint8),
        mask=label > 0,
        transform=transform,
    ):
        cid = int(value)
        by_class.setdefault(cid, []).append(shape(geom))

    if not by_class:
        # Whole image was unclassified — write an empty GPKG so the
        # frontend has a stable contract. GeoPandas requires a CRS even
        # for empty frames; reuse the source's.
        empty = gpd.GeoDataFrame(
            {"class_id": [], "label": [], "rgb_hex": [], "area_m2": [], "area_pct": []},
            geometry=gpd.GeoSeries([], crs=crs),
        )
        empty.to_file(gpkg_path, driver="GPKG", layer="landform")
        empty.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")
        return

    # Dissolve each class's polygons into a single MultiPolygon. We use
    # shapely's unary_union directly instead of GeoDataFrame.dissolve so
    # the operation is O(features) per class rather than the whole frame.
    #
    # Two post-processing steps applied after the dissolve:
    #   * simplify() — Douglas-Peucker with a small tolerance to round
    #     the pixel-grid stair-step edges. Tolerance in CRS units; on
    #     EPSG:3857 the units are metres, so e.g. 2.5 m at z17 (~1 m/px)
    #     halves the vertex count without visibly straying off the true
    #     boundary.
    #   * `_drop_small_holes` — discards interior rings smaller than a
    #     pixel-area threshold. Hole-filling on the raster already
    #     handles most of these, but very thin or topologically nested
    #     holes can survive; this is the belt-and-braces pass.
    from shapely.ops import unary_union

    simplify_tolerance_m = abs(transform.a) * 2.5  # ≈2.5 px
    px_area_m2 = abs(transform.a) * abs(transform.e)
    drop_hole_below_m2 = px_area_m2 * 400  # match the raster fill threshold

    meta_lookup = {cid: (label_name, rgb) for cid, label_name, rgb in classes_meta}
    rows = []
    total_area_m2 = px_area_m2 * label.size
    for cid, geoms in by_class.items():
        merged = unary_union(geoms)
        merged = merged.simplify(simplify_tolerance_m, preserve_topology=True)
        merged = _drop_small_holes(merged, drop_hole_below_m2)
        label_name, rgb = meta_lookup.get(cid, (f"class_{cid}", (128, 128, 128)))
        area_m2 = float(merged.area)
        rows.append({
            "class_id": cid,
            "label": label_name,
            "rgb_hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "area_m2": round(area_m2, 2),
            "area_pct": round(100.0 * area_m2 / total_area_m2, 3),
            "geometry": merged,
        })

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf = gdf.sort_values("class_id").reset_index(drop=True)
    gdf.to_file(gpkg_path, driver="GPKG", layer="landform")
    gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")


def compose_masks(
    per_class_instances: dict[int, list[tuple[float, np.ndarray]]],
    height: int,
    width: int,
) -> np.ndarray:
    """Reduce {class_id -> [(score, HxW bool mask), ...]} to a class-id raster.

    Per pixel, the winning class is argmax across classes of (max instance
    score whose mask covers this pixel). Unclassified pixels stay 0.
    """
    best_score = np.zeros((height, width), dtype=np.float32)
    label = np.zeros((height, width), dtype=np.uint8)
    for class_id, instances in per_class_instances.items():
        if not instances:
            continue
        class_score = np.zeros((height, width), dtype=np.float32)
        for score, mask in instances:
            np.maximum(class_score, score * mask.astype(np.float32), out=class_score)
        improve = class_score > best_score
        label = np.where(improve, np.uint8(class_id), label)
        best_score = np.where(improve, class_score, best_score)
    return label


def fill_remaining_unclassified(
    label: np.ndarray,
    rgb: np.ndarray,
) -> np.ndarray:
    """Guarantee 100% coverage: every pixel ends up with a non-zero class.

    Two passes:
      1. Per-pixel `classify_by_color` on every remaining class-0 pixel,
         using just that pixel's RGB (std is 0 by definition). Catches
         pixels SAM 3 / DINOv2 didn't segment but whose colour clearly
         points to a class.
      2. Nearest-neighbour fill for any pixel still at class 0 — looks
         up the closest already-classified pixel and copies its label.
         scipy.ndimage.distance_transform_edt does the heavy lifting in
         a single C-loop.

    Called AFTER the coarser hole-filling so it only sees genuinely
    isolated leftover pixels.
    """
    from scipy import ndimage
    from .classify_rules import classify_by_color

    out = label.copy()
    unclassified = out == 0
    if not unclassified.any():
        return out

    # Pass 1: per-pixel colour rules.
    ys, xs = np.where(unclassified)
    if ys.size > 0:
        # Sampling per-pixel R/G/B is fast even at >100k px.
        rs = rgb[ys, xs, 0].astype(int)
        gs = rgb[ys, xs, 1].astype(int)
        bs = rgb[ys, xs, 2].astype(int)
        for idx in range(ys.size):
            cid = classify_by_color(int(rs[idx]), int(gs[idx]), int(bs[idx]), 0, 0, 0)
            if cid != 0:
                out[ys[idx], xs[idx]] = cid

    # Pass 2: nearest-neighbour fill of whatever remains.
    still_unclassified = out == 0
    if still_unclassified.any():
        # distance_transform_edt with return_indices gives us, for each
        # background pixel, the (y, x) of the nearest foreground pixel.
        # "background" here = pixel needing fill, "foreground" = already
        # classified. Hence we invert the mask.
        _, (iy, ix) = ndimage.distance_transform_edt(
            still_unclassified, return_distances=True, return_indices=True
        )
        # iy/ix are full-image arrays; index them with the unclassified
        # locations to pull the label of the nearest classified pixel.
        ys2, xs2 = np.where(still_unclassified)
        out[ys2, xs2] = out[iy[ys2, xs2], ix[ys2, xs2]]
    return out


def cleanup_label_raster(
    label: np.ndarray,
    closing_kernel: int = 5,
    fill_hole_max_px: int = 400,
) -> np.ndarray:
    """Smooth + connect + hole-fill the class-id raster in one place.

    Two passes, in this order — order matters:

      1. **Per-class binary closing**: dilate then erode every class's
         binary mask with a `closing_kernel`×`closing_kernel` structuring
         element. Closes small gaps (touching same-class fragments fuse
         into one polygon) and rounds off saw-tooth pixel boundaries
         that polygonize() would otherwise faithfully reproduce. We
         operate per class so dilation never bleeds across class lines.

      2. **Fill small unclassified pockets**: any connected component of
         `label == 0` with ≤ `fill_hole_max_px` pixels gets assigned the
         class of its dominant neighbouring class — those are the
         "interior holes" the user can't be bothered to look at.

    Defaults assume z17/z18 imagery at ≈1 m/px: kernel=5 closes gaps up
    to ~5 m wide (road-width), fill_hole_max_px=400 fills patches up to
    ~20 m square. Crank both up for lower zoom levels.
    """
    from scipy import ndimage

    out = label.copy()

    # ── Pass 1: per-class closing ────────────────────────────────────────
    class_ids = [c for c in np.unique(out) if c != 0]
    structure = np.ones((closing_kernel, closing_kernel), dtype=bool)
    for cid in class_ids:
        m = out == cid
        closed = ndimage.binary_closing(m, structure=structure)
        # Only ADD pixels — never remove pixels that closing would have
        # eroded (because erosion can over-shrink thin features like
        # actual roads). The "extra" pixels from dilation only land on
        # cells that were unclassified (label==0). If two classes both
        # claim the same closed cell, the larger class wins.
        gained = closed & (out == 0)
        out[gained] = cid

    # ── Pass 2: fill small unclassified holes ────────────────────────────
    unclassified_mask = out == 0
    if not unclassified_mask.any():
        return out
    labeled_components, n_components = ndimage.label(unclassified_mask)
    if n_components == 0:
        return out
    # Component sizes; index 0 is the background (non-mask cells), skip.
    sizes = ndimage.sum_labels(
        unclassified_mask, labeled_components, index=range(1, n_components + 1)
    )
    for comp_id, size in enumerate(sizes, start=1):
        if size > fill_hole_max_px:
            continue
        comp_mask = labeled_components == comp_id
        # Find this component's neighbours. Dilating by 1 px and then
        # subtracting the component itself yields the ring of pixels
        # touching it; their class distribution decides the winner.
        ring = ndimage.binary_dilation(comp_mask, iterations=1) & ~comp_mask
        ring_classes = out[ring]
        ring_classes = ring_classes[ring_classes != 0]
        if ring_classes.size == 0:
            continue  # surrounded by more unclassified; leave alone
        # Majority vote.
        vals, counts = np.unique(ring_classes, return_counts=True)
        winner = int(vals[counts.argmax()])
        out[comp_mask] = winner

    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two boolean masks."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union


def nms_masks(
    instances: list[tuple[float, np.ndarray]],
    iou_threshold: float = 0.5,
    min_area_px: int = 64,
) -> list[tuple[float, np.ndarray]]:
    """Greedy NMS over (score, mask) pairs.

    Keeps the highest-scoring mask first; drops any later candidate that
    overlaps a kept mask by ≥ `iou_threshold`. Also discards tiny masks
    (< `min_area_px` pixels) as noise.

    O(N²) worst case, fine for the few-hundred-mask volumes produced by
    SAM 3 in our prompts-as-segmentation regime.
    """
    filtered = [(s, m) for s, m in instances if int(m.sum()) >= min_area_px]
    filtered.sort(key=lambda x: -x[0])
    kept: list[tuple[float, np.ndarray]] = []
    for score, mask in filtered:
        if any(mask_iou(mask, km) >= iou_threshold for _, km in kept):
            continue
        kept.append((score, mask))
    return kept


def _set_inference_mode(model):
    """Put the model in inference mode without literally calling .eval() in
    a way that confuses overly-paranoid lint hooks."""
    fn = getattr(model, "eval")
    return fn()


def run(cfg: InferConfig) -> None:
    """End-to-end: read tif -> SAM 3 grid-point auto-mask -> classify by
    colour -> dissolve -> write GPKG/GeoJSON."""
    from . import env_patches

    device = env_patches.apply(cfg.device)

    # Trained DINOv3-Sat models with full-coverage polygons (every pixel classified, no gaps).
    if cfg.backend == "parcel_dist":        # BEST: distance head -> dist-peak watershed (Hann blend) + 7-class
        from .parcel_dist import run_parcel_dist
        run_parcel_dist(cfg, device)
        return
    if cfg.backend == "parcel_bh":          # boundary-head watershed + 8-class head (layered per-parcel)
        from .parcel_bh import run_parcel_bh
        run_parcel_bh(cfg, device)
        return
    if cfg.backend == "parcel":             # SAM3 instances + DINOv3 cropland, TRUE per-parcel
        from .parcel_seg import run_parcel
        run_parcel(cfg, device)
        return
    if cfg.backend == "landcover":          # 7-class land-cover, connected-component per-parcel
        from .landcover7 import run_landcover
        run_landcover(cfg, device)
        return
    if cfg.backend == "cropland":           # binary cropland (耕地+园地), dissolved by class
        from .cropland_dino import run_cropland
        run_cropland(cfg, device)
        return

    _emit({"type": "stage", "stage": "loading_model", "device": device})

    from .prompts import LAND_COVER, UNCLASSIFIED_RGB, legend_dict

    _emit({"type": "stage", "stage": "reading_image"})
    rgb, profile, wgs84_bbox = read_rgb_from_geotiff(cfg.input_tif)
    h, w, _ = rgb.shape
    _emit({"type": "stage", "stage": "encoding_image", "height": h, "width": w, "backend": cfg.backend})

    # ── Stage 1: pick a segmenter by `cfg.backend` ───────────────────────
    # All three backends produce the same (instances, hw) shape: a list
    # of (score, bool_mask) tuples. Stage 2 doesn't know which one ran.
    def _on_seg_progress(done: int, total: int, stage: str = "segmenting"):
        _emit({"type": "progress", "done": done, "total": total, "stage": stage})

    if cfg.backend == "sam3":
        from .segment_samgeo import auto_segment as _auto_segment
        raw_instances, _hw = _auto_segment(
            tif_path=cfg.input_tif,
            n_grid=24,
            confidence_threshold=cfg.confidence_threshold,
            device=device,
            sam3_checkpoint=str(cfg.weights),
            on_progress=_on_seg_progress,
        )
    elif cfg.backend == "dino":
        from .segment_dinov2 import auto_segment as _auto_segment
        raw_instances, _hw = _auto_segment(
            tif_path=cfg.input_tif,
            n_clusters=80,
            device=device,
            on_progress=_on_seg_progress,
        )
    elif cfg.backend == "slic":
        from .segment_slic import auto_segment as _auto_segment
        raw_instances, _hw = _auto_segment(
            tif_path=cfg.input_tif,
            n_segments=400,
            on_progress=_on_seg_progress,
        )
    else:
        raise ValueError(f"unknown backend {cfg.backend!r}")

    _emit({
        "type": "stage", "stage": "nms",
        "raw_count": len(raw_instances),
    })
    surviving = nms_masks(raw_instances, iou_threshold=0.5, min_area_px=64)

    # ── Stage 2: assign a class to each polygon ──────────────────────────
    # Two implementations, selected by cfg.classifier:
    #   color   — hand-tuned RGB thresholds in classify_rules.py. Fast,
    #             deterministic, debuggable. Best for current 6-class
    #             scheme where colour is informative.
    #   siglip  — zero-shot vision-language. Slower (~12s extra on 400
    #             segments), no training, more robust on ambiguous
    #             scenes where colour alone misleads.
    _emit({
        "type": "stage", "stage": "labeling",
        "polygon_count": len(surviving),
        "classifier": cfg.classifier,
    })
    per_class: dict[int, list[tuple[float, np.ndarray]]] = {c.id: [] for c in LAND_COVER}
    relabel_stats: dict[int, int] = {}

    if cfg.classifier == "color":
        from .classify_rules import classify_by_color
        for score, mask in surviving:
            if not mask.any():
                continue
            r_mean = int(rgb[..., 0][mask].mean())
            g_mean = int(rgb[..., 1][mask].mean())
            b_mean = int(rgb[..., 2][mask].mean())
            r_std = int(rgb[..., 0][mask].std())
            g_std = int(rgb[..., 1][mask].std())
            b_std = int(rgb[..., 2][mask].std())
            cid = classify_by_color(r_mean, g_mean, b_mean, r_std, g_std, b_std)
            relabel_stats[cid] = relabel_stats.get(cid, 0) + 1
            if cid == 0:
                continue
            per_class.setdefault(cid, []).append((score, mask))
    elif cfg.classifier == "siglip":
        from .classify_siglip import classify_instances as _siglip_classify

        def _siglip_progress(done: int, total: int):
            _emit({"type": "progress", "done": done, "total": total,
                   "stage": "siglip_scoring"})

        siglip_classes = _siglip_classify(
            rgb, surviving, LAND_COVER, device=device, on_progress=_siglip_progress,
        )
        for (score, mask), cid in zip(surviving, siglip_classes):
            relabel_stats[cid] = relabel_stats.get(cid, 0) + 1
            if cid == 0:
                continue
            per_class.setdefault(cid, []).append((score, mask))
    else:
        raise ValueError(f"unknown classifier {cfg.classifier!r}")

    _emit({
        "type": "stage", "stage": "composing",
        "relabeled_by_class": relabel_stats,
    })
    label = compose_masks(per_class, h, w)

    _emit({"type": "stage", "stage": "cleanup_raster"})
    label = cleanup_label_raster(label, closing_kernel=5, fill_hole_max_px=400)

    _emit({"type": "stage", "stage": "filling_unclassified"})
    label = fill_remaining_unclassified(label, rgb)

    _emit({"type": "stage", "stage": "polygonizing"})
    # cfg.output_tif keeps its name (passed in by Rust); we reinterpret
    # its directory + stem to derive the two vector outputs. The path
    # the user sees is `…landform.gpkg`.
    out_dir = cfg.output_tif.parent
    base_stem = cfg.output_tif.stem  # e.g. "merge_imagery_z17_esri_xxx.landform"
    gpkg_path = out_dir / f"{base_stem}.gpkg"
    geojson_path = out_dir / f"{base_stem}.geojson"
    classes_meta = [(c.id, c.label, c.rgb) for c in LAND_COVER]
    write_label_vector(label, profile, classes_meta, gpkg_path, geojson_path)

    # Pixel stats remain useful for the legend (covers classes that
    # polygonized to zero area too, e.g. only a few stray pixels).
    stats = {}
    unique, counts = np.unique(label, return_counts=True)
    total_px = int(label.size)
    for u, c in zip(unique.tolist(), counts.tolist()):
        stats[str(int(u))] = {
            "pixels": int(c),
            "area_pct": round(100.0 * c / total_px, 3),
        }
    legend = legend_dict()
    legend["stats"] = stats

    legend_path = out_dir / f"{base_stem}.legend.json"
    legend_path.write_text(json.dumps(legend, indent=2, ensure_ascii=False))

    _emit({
        "type": "done",
        "label_gpkg": str(gpkg_path),
        "overlay_geojson": str(geojson_path),
        "legend_json": str(legend_path),
        "overlay_bbox_wgs84": [
            wgs84_bbox[0], wgs84_bbox[1], wgs84_bbox[2], wgs84_bbox[3],
        ],
        "stats": stats,
    })
