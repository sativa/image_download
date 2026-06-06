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


def build_idmap(clsprob, dist, bnd, a):
    """Full-coverage instance id-map: cropland(1,2) via dist-peak watershed; other classes via
    connected components of the per-pixel argmax. Returns (idmap, cls_of)."""
    pix = clsprob[1:].argmax(0) + 1                                # 1..8 per pixel
    for s, d in MERGE.items():
        pix[pix == s] = d                                          # 8->6
    # cropland/orchard instances (the well-delineated part)
    inst, n = dist_peak_instances(clsprob, dist, bnd, a.min_dist, a.peak_thr, a.min_area_px, a.ridge)
    idmap = inst.astype(np.int32); cls_of = {}
    flat = clsprob[1:].reshape(8, -1)
    for pid in range(1, n + 1):
        m = idmap == pid
        if m.any():
            cls_of[pid] = int(flat[:, m.ravel()].mean(1).argmax()) + 1   # 1 or 2 (crop/orchard)
    nxt = n + 1
    # other land-cover classes: connected components of argmax in still-unassigned pixels
    for c in (3, 4, 5, 6, 7):
        cm = ((idmap == 0) & (pix == c)).astype(np.uint8)
        if not cm.any():
            continue
        ncc, cc = cv2.connectedComponents(cm, connectivity=8)
        for lab in range(1, ncc):
            region = cc == lab
            if int(region.sum()) < a.min_area_px:
                continue
            idmap[region] = nxt; cls_of[nxt] = c; nxt += 1
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
            gs = g.simplify(simp, preserve_topology=True)
            g = gs if (not gs.is_empty and gs.is_valid) else g
        area_m2 = float(areas_px[pid]) * pix_m * pix_m if pid < len(areas_px) else 0.0
        rows.append({"parcel_id": pid, "class_id": c, "label": NAME_ZH[c], "label_en": NAME_EN[c],
                     "rgb_hex": HEX[c], "area_m2": round(area_m2, 1), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gpkg = out_dir / f"{name}.gpkg"; geojson = out_dir / f"{name}.geojson"
    gdf.to_file(gpkg, driver="GPKG", layer="parcels")
    gdf.to_file(geojson, driver="GeoJSON")
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
    return len(rows), {NAME_ZH[k]: v for k, v in sorted(cc.items())}


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


def main():
    import torch
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="/mnt/sda/zf/landform/results/parcel_product")
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
    else:
        manf = Path(a.data_dir) / "manifest.json"
        mm = json.loads(manf.read_text()) if manf.exists() else {}
        pool = mm.get("test") or sorted(p.stem for p in Path(a.data_dir).glob("*.npz"))
        cells = [n for n in pool if (Path(a.data_dir) / f"{n}.npz").exists()][:a.n_cells]
    for n in cells:
        z = np.load(Path(a.data_dir) / f"{n}.npz")
        npar, stats = export_cell(m, z["x6"], z["bbox"], n, a, out_dir)
        print(f"  {n}: {npar} parcels  {stats}  -> {n}.geojson/.gpkg/.png", flush=True)
    print(f"[export] done -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
