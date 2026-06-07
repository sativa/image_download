"""Export per-parcel land-cover PRODUCT (vector polygons + class) using the best delineation line:
DINOv3-Sat distance head -> dist-peak watershed instances for cropland, classifier-argmax connected
components for the other classes -> full-coverage 7-class instance map -> GeoJSON (EPSG:4326) + GPKG +
a PNG (RGB | classified | instance edges) + legend.json. This is the deployable "分类图+矢量地块"
deliverable from the tuned dist-peak route (area-match 98%). Run on .250."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import cv2
import rasterio.features
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from dino_parcel_eval import infer_heads, dist_peak_instances  # reuse inference

CLASSES = [(1, "耕地", "cropland", (60, 180, 75)), (2, "园地", "orchard", (170, 255, 90)),
           (3, "林地", "forest", (0, 100, 0)), (4, "草地", "grassland", (190, 220, 100)),
           (5, "水体", "water", (0, 130, 200)), (6, "建筑", "built-up", (230, 25, 75)),
           (7, "荒漠", "bare", (170, 140, 100))]
MERGE = {8: 6}  # 设施大棚 -> 建筑
NAME_ZH = {c[0]: c[1] for c in CLASSES}; NAME_EN = {c[0]: c[2] for c in CLASSES}
RGB = {c[0]: c[3] for c in CLASSES}; HEX = {c[0]: "#%02x%02x%02x" % c[3] for c in CLASSES}


def smooth_geom(geom, iters=2):
    """Chaikin corner-cutting: rasterised parcel edges (pixel staircase) -> smooth curves. Topology-safe."""
    from shapely.geometry import Polygon, MultiPolygon

    def chaikin(coords):
        pts = list(coords)
        if len(pts) < 4:
            return pts
        for _ in range(iters):
            out = []
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                out.append((0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]))
                out.append((0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]))
            out.append(out[0]); pts = out
        return pts

    try:
        if geom.geom_type == "Polygon":
            g2 = Polygon(chaikin(geom.exterior.coords), [chaikin(r.coords) for r in geom.interiors])
            return g2 if g2.is_valid else geom
        if geom.geom_type == "MultiPolygon":
            mp = MultiPolygon([smooth_geom(p, iters) for p in geom.geoms])
            return mp if mp.is_valid else geom
    except Exception:
        return geom
    return geom


def build_idmap(clsprob, dist, bnd, a):
    """Full-coverage instance id-map: cropland(1,2) via dist-peak watershed; other classes via
    connected components of the per-pixel argmax. Returns (idmap, cls_of)."""
    pix = clsprob[1:].argmax(0) + 1                                # 1..8 per pixel
    for s, d in MERGE.items():
        pix[pix == s] = d                                          # 8->6
    # cropland/orchard instances (the well-delineated part); downscale lets a big mosaic run in one pass
    inst, n = dist_peak_instances(clsprob, dist, bnd, a.min_dist, a.peak_thr, a.min_area_px,
                                  a.ridge, getattr(a, "downscale", 1))
    idmap = inst.astype(np.int32); cls_of = {}
    if n > 0:                                                      # vectorised cls per instance (bincount, no per-pid mask)
        fid = idmap.ravel()
        s1 = np.bincount(fid, weights=clsprob[1].ravel().astype(np.float64), minlength=n + 1)
        s2 = np.bincount(fid, weights=clsprob[2].ravel().astype(np.float64), minlength=n + 1)
        for pid in range(1, n + 1):
            cls_of[pid] = 1 if s1[pid] >= s2[pid] else 2           # cropland(1) vs orchard(2)
    nxt = n + 1
    # other land-cover classes: connected components of argmax in still-unassigned pixels
    for c in (3, 4, 5, 6, 7):
        cm = ((idmap == 0) & (pix == c)).astype(np.uint8)
        if not cm.any():
            continue
        ncc, cc, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)   # vectorised, no per-label mask scan
        remap = np.zeros(ncc, np.int32)
        for lab in range(1, ncc):
            if stats[lab, cv2.CC_STAT_AREA] >= a.min_area_px:
                remap[lab] = nxt; cls_of[nxt] = c; nxt += 1
        sel = cc > 0
        gid = remap[cc[sel]]
        idmap[sel] = np.where(gid > 0, gid, idmap[sel])            # assign whole class in one vectorised step
    return idmap, cls_of


def export_cell(model, x6, bbox, name, a, out_dir):
    import math
    _, H, W = x6.shape
    clsprob, dist, bnd = infer_heads(model, x6, a.device)
    idmap, cls_of = build_idmap(clsprob, dist, bnd, a)
    transform = from_bounds(*[float(b) for b in bbox], W, H)
    pix_m = (float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(
        math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W
    simp = a.simplify_px * abs(transform.a)
    areas_px = np.bincount(idmap.ravel())                          # px per parcel -> physical m^2 (4326 area is deg^2)
    rows = []
    for geom, val in rasterio.features.shapes(idmap, mask=idmap > 0, connectivity=8, transform=transform):
        pid = int(val); c = cls_of.get(pid)
        if not c:
            continue
        g = shape(geom)
        if simp > 0:
            gs = g.simplify(simp, preserve_topology=True)          # drop pixel-staircase points
            g = gs if (not gs.is_empty and gs.is_valid) else g
        if getattr(a, "smooth_iters", 2) > 0:
            g = smooth_geom(g, a.smooth_iters)                     # Chaikin -> smooth curved edges
        area_m2 = float(areas_px[pid]) * pix_m * pix_m if pid < len(areas_px) else 0.0
        rows.append({"parcel_id": pid, "class_id": c, "label": NAME_ZH[c], "label_en": NAME_EN[c],
                     "rgb_hex": HEX[c], "area_m2": round(area_m2, 1), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["cell"] = name
    gdf.to_parquet(out_dir / f"{name}.parquet")                    # GeoParquet (default storage)
    if not a.parquet_only:
        gdf.to_file(out_dir / f"{name}.gpkg", driver="GPKG", layer="parcels")
        gdf.to_file(out_dir / f"{name}.geojson", driver="GeoJSON")
    # raster + viz
    clsmap = np.zeros((H, W), np.uint8)
    for pid, c in cls_of.items():
        clsmap[idmap == pid] = c
    _viz(x6, idmap, clsmap, out_dir / f"{name}.png")
    from collections import Counter
    cc = Counter(r["class_id"] for r in rows)
    legend = {"classes": [{"id": c[0], "label": c[1], "label_en": c[2], "rgb": list(c[3])} for c in CLASSES],
              "model": "DINOv3-Sat dist-peak watershed (cropland) + 7-class connected-components",
              "n_parcels": len(rows), "stats": {NAME_ZH[k]: v for k, v in cc.items()}}
    (out_dir / f"{name}.legend.json").write_text(json.dumps(legend, indent=2, ensure_ascii=False))
    return gdf, len(rows), {NAME_ZH[k]: v for k, v in sorted(cc.items())}


def _viz(x6, idmap, clsmap, path, max_side=1100):
    H, W = idmap.shape; s = max(1, max(H, W) // max_side)
    rgb = np.ascontiguousarray(x6[:3].transpose(1, 2, 0))[::s, ::s].astype(np.uint8)
    cm = clsmap[::s, ::s]; idm = idmap[::s, ::s]
    # classified overlay
    col = np.zeros((*cm.shape, 3), np.uint8)
    for cid, rgbv in RGB.items():
        col[cm == cid] = rgbv
    clsov = (0.45 * rgb + 0.55 * col).astype(np.uint8)
    # instance edges (parcel boundaries) on RGB
    edge = np.zeros(idm.shape, bool)
    edge[:-1, :] |= idm[:-1, :] != idm[1:, :]; edge[:, :-1] |= idm[:, :-1] != idm[:, 1:]
    edgeov = rgb.copy(); edgeov[edge & (idm > 0)] = (255, 255, 0)
    gap = np.full((rgb.shape[0], 8, 3), 255, np.uint8)
    from PIL import Image
    Image.fromarray(np.concatenate([rgb, gap, clsov, gap, edgeov], 1)).save(path)


def load_tif_pair(tif_dir, cell):
    """Route 2 (local imagery): read {cell}_esri.tif + {cell}_google.tif -> x6 (6,H,W) uint8 + bbox(4326).
    Google resized to Esri grid if sizes differ; bbox from Esri CRS reprojected to WGS84."""
    import rasterio
    from rasterio.warp import transform_bounds
    tp = Path(tif_dir)
    with rasterio.open(tp / f"{cell}_esri.tif") as e:
        re = e.read([1, 2, 3]); crs = e.crs; b = e.bounds; H, W = e.height, e.width
    gp = tp / f"{cell}_google.tif"
    if gp.exists():
        with rasterio.open(gp) as g:
            rg = g.read([1, 2, 3])
        if rg.shape[1:] != (H, W):
            rg = np.stack([cv2.resize(rg[i], (W, H), interpolation=cv2.INTER_LINEAR) for i in range(3)])
    else:
        rg = re                                                    # google missing -> duplicate esri
    x6 = np.concatenate([re, rg], 0).astype(np.uint8)
    bbox = np.array(transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top), np.float64)
    return x6, bbox


def main():
    import torch
    import pandas as pd
    from transformers import AutoModel
    from train_dino_1m_v3 import DinoV3FreqUNetBDD, DINOV3_SAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_v3_bdd/best.pt")
    ap.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_lc")
    ap.add_argument("--cells", default="", help="comma cell names; empty=first N test cells")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--min-dist", type=int, default=20)
    ap.add_argument("--peak-thr", type=float, default=0.4)
    ap.add_argument("--min-area-px", type=int, default=200)
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--simplify-px", type=float, default=2.0)
    ap.add_argument("--downscale", type=int, default=1, help=">1: watershed on /N grid (big mosaic in one pass, no cell seams)")
    ap.add_argument("--smooth-iters", type=int, default=2, help="Chaikin corner-cutting iterations (0=off)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="/mnt/sda/zf/landform/results/parcel_product")
    ap.add_argument("--prefix", default="", help="glob cells by prefix (e.g. 620123_ for a whole county)")
    ap.add_argument("--tif-dir", default="", help="route 2: local esri+google tif dir (else npz data-dir)")
    ap.add_argument("--region-out", default="", help="merge all cells -> one regional GeoParquet")
    ap.add_argument("--parquet-only", action="store_true", help="skip per-cell gpkg/geojson, keep GeoParquet")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBDD(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(a.device)
    sd = torch.load(a.ckpt, map_location=a.device, weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    print(f"[export] loaded {a.ckpt} ({time.time()-t0:.0f}s)", flush=True)
    if a.cells:
        cells = a.cells.split(",")
    elif a.tif_dir:                                                # route 2: pair from *_esri.tif
        cells = sorted(p.name[:-9] for p in Path(a.tif_dir).glob(f"{a.prefix}*_esri.tif"))
    elif a.prefix:                                                 # whole county from npz dir
        cells = sorted(p.stem for p in Path(a.data_dir).glob(f"{a.prefix}*.npz"))
    else:
        manf = Path(a.data_dir) / "manifest.json"
        mm = json.loads(manf.read_text()) if manf.exists() else {}
        pool = mm.get("test") or sorted(p.stem for p in Path(a.data_dir).glob("*.npz"))
        cells = [n for n in pool if (Path(a.data_dir) / f"{n}.npz").exists()][:a.n_cells]
    if a.n_cells and len(cells) > a.n_cells and not a.cells:
        cells = cells[:a.n_cells]
    print(f"[export] {len(cells)} cells | route={'local-tif' if a.tif_dir else 'download-npz'}", flush=True)
    gdfs = []
    for n in cells:
        try:
            if a.tif_dir:
                x6, bbox = load_tif_pair(a.tif_dir, n)
            else:
                z = np.load(Path(a.data_dir) / f"{n}.npz"); x6, bbox = z["x6"], z["bbox"]
            gdf, npar, stats = export_cell(m, x6, bbox, n, a, out_dir)
            gdfs.append(gdf)
            print(f"  {n}: {npar} parcels  {stats}", flush=True)
        except Exception as ex:
            print(f"  {n}: FAILED {ex}", flush=True)
    if a.region_out and gdfs:
        reg = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
        reg.insert(0, "gid", range(1, len(reg) + 1))               # global region id
        reg.to_parquet(a.region_out)
        from collections import Counter
        cc = Counter(reg["label"])
        print(f"[region] {len(reg)} parcels -> {a.region_out}  {dict(cc)}", flush=True)
    print(f"[export] done -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
