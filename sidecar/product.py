"""Combined product: DINOv2-1m (classification map) + SAM3 (vector parcels) -> classified parcels.

For each cell:
  - DINOv2-1m -> per-pixel cropland probability (the strong 0.86/0.843 classifier).
  - SAM3 (fine-tuned if available) -> instance parcel masks ("crop field"), tiled.
  - Each SAM3 parcel gets cropland class = (mean DINO prob under it > 0.5) -> CLASSIFIED VECTOR PARCELS.
Outputs per cell: <name>.geojson (parcel polygons + class + dino_conf, EPSG:4326),
  <name>_cropland.tif (the 1m DINO cropland map, EPSG:4326), <name>_product.png (viz).
"""
import argparse, json, sys, time, types
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

SAM3_REPO = "/home/ps/sam3/sam3-inference"
sys.path.insert(0, SAM3_REPO)
try:
    import decord  # noqa
except Exception:
    m = types.ModuleType("decord"); m.cpu = m.gpu = lambda *a, **k: None
    m.VideoReader = object; m.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
    sys.modules["decord"] = m

import rasterio.features
from rasterio.transform import from_bounds, Affine
from shapely.geometry import shape, mapping

DEV = "cuda"


@torch.no_grad()
def dino_cropland_prob(model, x6, cs=448):
    _, SZ, SZw = x6.shape
    acc = np.zeros((3, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xb = torch.from_numpy(norm6(x6[:, t:t + cs, l:l + cs])).unsqueeze(0).to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(xb)
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
    acc /= np.maximum(cnt, 1)
    return acc[1] / np.maximum(acc[1] + acc[2], 1e-6)  # cropland prob (vs other)


def load_sam3(weights, head_state):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=weights, load_from_HF=False, device=DEV)
    if head_state and Path(head_state).exists():
        st = torch.load(head_state, map_location=DEV, weights_only=True)  # own file: dict of state_dicts (tensors)
        if isinstance(st, dict) and "seg" in st:  # full ft_state.pt
            model.segmentation_head.load_state_dict(st["seg"])
            model.transformer.load_state_dict(st["trans"])
            if st.get("dps") is not None and getattr(model, "dot_prod_scoring", None) is not None:
                model.dot_prod_scoring.load_state_dict(st["dps"])
            print(f"[product] loaded FULL fine-tuned SAM3 ({head_state})", flush=True)
        else:  # seg_head.pt only
            model.segmentation_head.load_state_dict(st)
            print(f"[product] loaded fine-tuned seg_head ({head_state})", flush=True)
    else:
        print("[product] SAM3 zero-shot (no fine-tuned head)", flush=True)
    return Sam3Processor(model, device=DEV, confidence_threshold=0.4)


def parcels(proc, rgb, dino_p, prompt, tile, transform):
    """SAM3 instance masks (tiled) -> classified geo polygons (class from DINO prob under each)."""
    H, W = rgb.shape[:2]; feats = []
    wins = [(y, x, min(y + tile, H), min(x + tile, W)) for y in range(0, H, tile) for x in range(0, W, tile)] if tile else [(0, 0, H, W)]
    for (y0, x0, y1, x1) in wins:
        st = proc.set_image(Image.fromarray(rgb[y0:y1, x0:x1]))
        st = proc.set_text_prompt(prompt=prompt, state=st)
        mk = st.get("masks")
        if mk is None or mk.numel() == 0:
            continue
        wtrans = transform * Affine.translation(x0, y0)
        for inst in mk.squeeze(1).cpu().numpy().astype(np.uint8):
            if inst.sum() < 200:
                continue
            sub = dino_p[y0:y1, x0:x1][inst > 0]
            conf = float(sub.mean()) if sub.size else 0.0
            cls = "cropland" if conf > 0.5 else "other"
            for geom, _ in rasterio.features.shapes(inst, mask=inst > 0, transform=wtrans):
                poly = shape(geom).simplify(2 * abs(transform.a), preserve_topology=True)
                if poly.area > 0:
                    feats.append({"type": "Feature", "geometry": mapping(poly),
                                  "properties": {"class": cls, "dino_conf": round(conf, 3)}})
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dino-ckpt", default="/mnt/sda/zf/landform/results/dino_1m/best.pt")
    p.add_argument("--sam3-weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--sam3-head", default="/mnt/sda/zf/landform/results/sam3_ft_b/ft_state.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--n-cells", type=int, default=3)
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/product")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    dino = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=4).to(DEV)
    dino.load_state_dict(torch.load(a.dino_ckpt, map_location=DEV, weights_only=True)); dino.eval()
    head = a.sam3_head if Path(a.sam3_head).exists() else "/mnt/sda/zf/landform/results/sam3_ft_b/seg_head.pt"
    proc = load_sam3(a.sam3_weights, head)
    print(f"[product] models loaded ({time.time()-t0:.0f}s)", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    names = names[::max(1, len(names) // a.n_cells)][:a.n_cells]
    for name in names:
        z = np.load(Path(a.data_dir) / f"{name}.npz"); x6 = z["x6"]; bbox = z["bbox"]; H, W = x6.shape[1:]
        rgb = np.ascontiguousarray(x6[0:3].transpose(1, 2, 0))
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)
        dp = dino_cropland_prob(dino, x6)
        feats = parcels(proc, rgb, dp, a.prompt, a.tile, transform)
        nc = sum(1 for f in feats if f["properties"]["class"] == "cropland")
        (out / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        # cropland GeoTIFF (1m DINO map)
        try:
            import rasterio
            with rasterio.open(out / f"{name}_cropland.tif", "w", driver="GTiff", height=H, width=W,
                               count=1, dtype="uint8", crs="EPSG:4326", transform=transform) as ds:
                ds.write((dp > 0.5).astype(np.uint8), 1)
        except Exception as e:
            print(f"  tif skip: {e}", flush=True)
        # viz: RGB + parcels (green=cropland, red=other)
        s = max(1, H // 900); im = Image.fromarray(rgb[::s, ::s]); dr = ImageDraw.Draw(im)
        for f in feats:
            col = (0, 220, 0) if f["properties"]["class"] == "cropland" else (220, 0, 0)
            geom = shape(f["geometry"])
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for pp in polys:
                xs, ys = pp.exterior.xy
                pix = [(~transform * (x, y)) for x, y in zip(xs, ys)]
                dr.line([(px / s, py / s) for px, py in pix], fill=col, width=2)
        im.save(out / f"{name}_product.png")
        print(f"  {name}: {len(feats)} parcels ({nc} cropland) -> geojson+tif+png", flush=True)
    print(f"[product] done -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
