"""Demo: turn the fine-tuned DINOv2-1m cropland prediction into VECTOR polygons.

Answers "can DINO give vector polygons?" — YES. Predict cropland at 1m -> rasterio.features.shapes
polygonizes the mask -> Shapely Douglas-Peucker simplify -> GeoJSON (lon/lat via the cell bbox).
This yields cropland-REGION polygons (adjacent fields merge); per-PARCEL instances need a boundary
head + watershed (the standard field-delineation recipe) — noted as the next step.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from dino_changzhi_eval import predict_1m

import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m/best.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--n-cells", type=int, default=3)
    p.add_argument("--min-area-px", type=int, default=400, help="drop polygons < this many 1m px (<0.04 ha)")
    p.add_argument("--simplify-px", type=float, default=3.0)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_vec")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=4).to(a.device)
    model.load_state_dict(torch.load(a.ckpt, map_location=a.device, weights_only=True)); model.eval()

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    names = names[::max(1, len(names) // a.n_cells)][:a.n_cells]

    for name in names:
        z = np.load(Path(a.data_dir) / f"{name}.npz"); x6 = z["x6"]; bbox = z["bbox"]; lbl = z["label"]
        H, W = x6.shape[1:]
        pred = (predict_1m(model, x6, a.device, 448) == 1).astype(np.uint8)
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)  # 1m grid over the cell bbox (EPSG:4326)

        feats = []; px_polys = []
        for geom, val in rasterio.features.shapes(pred, mask=pred > 0, transform=transform):
            poly = shape(geom)
            # area in pixels via the un-transformed footprint: approximate by geo-area scaled back
            poly_s = poly.simplify(a.simplify_px * abs(transform.a), preserve_topology=True)
            feats.append({"type": "Feature", "properties": {"class": "cropland"}, "geometry": mapping(poly_s)})
        # pixel-space polygons for area filter + viz (identity transform)
        for geom, val in rasterio.features.shapes(pred, mask=pred > 0):
            poly = shape(geom)
            if poly.area >= a.min_area_px:
                px_polys.append(poly.simplify(a.simplify_px, preserve_topology=True))

        gj = {"type": "FeatureCollection", "features": feats}
        (out / f"{name}.geojson").write_text(json.dumps(gj))

        # viz: RGB (downsampled) with polygon outlines
        s = max(1, H // 900); rgb = np.ascontiguousarray(x6[0:3, ::s, ::s].transpose(1, 2, 0))
        im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
        for poly in px_polys:
            xs, ys = poly.exterior.xy
            dr.line([(x / s, y / s) for x, y in zip(xs, ys)], fill=(255, 0, 0), width=2)
        im.save(out / f"{name}_polys.png")

        crop_frac = (lbl == 1).sum() / max((lbl > 0).sum(), 1)
        print(f"  {name}: {len(px_polys)} cropland polygons (>{a.min_area_px}px), "
              f"cell crop={crop_frac*100:.0f}% -> {name}.geojson + {name}_polys.png", flush=True)
    print(f"[dino-vec] done -> {out}  (region polygons; per-parcel needs boundary-head+watershed)", flush=True)


if __name__ == "__main__":
    main()
