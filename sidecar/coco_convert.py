"""Convert c_1m (1m Esri RGB) + DLTB cropland parcels -> COCO dataset for SAM3 fine-tuning.

Each DLTB cropland polygon (耕地 cid=1 + 园地 cid=2) becomes an instance annotation; the category
name "crop field" is the text concept SAM3 learns. Cells are tiled into crops so individual parcels
are well-resolved. Output is the Roboflow/COCO layout the official sam3 trainer reads:
  <out>/{train,valid}/images/*.jpg  +  <out>/{train,valid}/_annotations.coco.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box as shp_box, Polygon, MultiPolygon
from shapely.affinity import affine_transform

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS

CONCEPT = "crop field"
SRC_MAP = {"esri": [(0, "esri")], "google": [(3, "google")], "both": [(0, "esri"), (3, "google")]}


def load_county(dltb_dir, county, cache):
    if county in cache:
        return cache[county]
    g = gpd.read_parquet(Path(dltb_dir) / f"{county}.parquet")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except AttributeError:
        g["geometry"] = g.geometry.buffer(0)
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
    g = g[g["cid"].isin([1, 2])].reset_index(drop=True)  # cropland = 耕地 + 园地
    cache[county] = g
    return g


def cell_pixel_polys(g, bbox, W, H):
    """Cropland polygons -> shapely polygons in full-cell pixel coords (origin top-left)."""
    w, s, e, n = bbox
    idx = list(g.sindex.intersection((w, s, e, n)))
    if not idx:
        return []
    sub = g.iloc[idx]
    a = W / (e - w); ey = -H / (n - s)            # px = a*lon + xoff ; py = ey*lat + yoff
    xoff = -w * a; yoff = -n * ey
    cb = shp_box(w, s, e, n)
    polys = []
    for geom in sub.geometry:
        gg = geom.intersection(cb)
        if gg.is_empty:
            continue
        polys.append(affine_transform(gg, [a, 0, 0, ey, xoff, yoff]))
    return polys


def poly_to_coco_segs(geom, T):
    """Polygon/MultiPolygon (crop-local px) -> (segmentation list, bbox, area), clipped to [0,T]."""
    geom = geom.intersection(shp_box(0, 0, T, T))
    if geom.is_empty:
        return None
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    segs = []; area = 0.0
    minx = miny = 1e9; maxx = maxy = -1e9
    for part in parts:
        if not isinstance(part, Polygon) or part.area < 1:
            continue
        xs, ys = part.exterior.coords.xy
        seg = []
        for x, y in zip(xs, ys):
            seg += [round(float(x), 1), round(float(y), 1)]
        segs.append(seg); area += part.area
        bx0, by0, bx1, by1 = part.bounds
        minx, miny, maxx, maxy = min(minx, bx0), min(miny, by0), max(maxx, bx1), max(maxy, by1)
    if not segs:
        return None
    return segs, [round(minx, 1), round(miny, 1), round(maxx - minx, 1), round(maxy - miny, 1)], area


def build_split(names, args, cache, split):
    out_img = Path(args.out) / split / "images"; out_img.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    img_id = 0; ann_id = 0
    for name in names:
        z = np.load(Path(args.data_dir) / f"{name}.npz")
        x6 = z["x6"]; bbox = z["bbox"]; H, W = x6.shape[1:]
        county = name.split("_")[0]
        try:
            g = load_county(args.dltb, county, cache)
        except Exception as ex:
            print(f"  skip {name}: {ex}", flush=True); continue
        cell_polys = cell_pixel_polys(g, bbox, W, H)
        if not cell_polys:
            continue
        T = args.tile
        for r0 in range(0, H, T):
            for c0 in range(0, W, T):
                r1, c1 = min(r0 + T, H), min(c0 + T, W)
                th, tw = r1 - r0, c1 - c0
                local = [affine_transform(p, [1, 0, 0, 1, -c0, -r0]) for p in cell_polys]
                segs_list = []
                for lp in local:
                    res = poly_to_coco_segs(lp, min(th, tw))
                    if res is None:
                        continue
                    segs, bx, area = res
                    if area < args.min_area:
                        continue
                    segs_list.append((segs, bx, area))
                if not segs_list:
                    continue  # skip empty crops (no cropland)
                for src, sname in SRC_MAP[args.sources]:  # one image per source (Esri/Google), shared parcels
                    fn = f"{name}_{r0}_{c0}_{sname}.jpg"
                    Image.fromarray(np.ascontiguousarray(x6[src:src + 3, r0:r1, c0:c1].transpose(1, 2, 0))).save(
                        out_img / fn, quality=92)
                    images.append({"id": img_id, "file_name": f"images/{fn}", "width": tw, "height": th})
                    for segs, bx, area in segs_list:
                        annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                            "segmentation": segs, "bbox": bx, "area": round(area, 1), "iscrowd": 0})
                        ann_id += 1
                    img_id += 1
        if img_id and img_id % 200 < args.tile // args.tile:  # light progress
            pass
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": CONCEPT, "supercategory": "land"}]}
    (Path(args.out) / split / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"[coco] {split}: {len(images)} crops, {len(annotations)} parcel instances", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--out", default="/mnt/sda/zf/landform/data/sam3_coco")
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--tile", type=int, default=1110)
    p.add_argument("--sources", choices=["esri", "google", "both"], default="both",
                   help="which 1m source(s) to emit as crops (both = dual-source fine-tune)")
    p.add_argument("--min-area", type=float, default=300.0, help="drop parcels < this many 1m px")
    a = p.parse_args()

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr = [n for n in man["train"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    tr = tr[::max(1, len(tr) // (a.n_train + a.n_val))]
    train_names, val_names = tr[:a.n_train], tr[a.n_train:a.n_train + a.n_val]
    print(f"[coco] tile={a.tile} train_cells={len(train_names)} val_cells={len(val_names)} concept='{CONCEPT}'", flush=True)
    cache = {}
    build_split(train_names, a, cache, "train")
    build_split(val_names, a, cache, "valid")
    print(f"[coco] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
