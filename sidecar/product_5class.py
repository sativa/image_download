"""Multi-class integrated product: DINOv2-1m 5-class landform map + SAM3 parcels -> 5-TYPE vector parcels.

Each SAM3 instance parcel is labeled with the majority DINO 5-class land type (耕地/园地/林地/草地/其他).
Outputs per cell: <name>.geojson (polygons + landform class + conf), <name>_landform.tif (5-class map),
<name>_5cls.png (parcels colored by type). Uses dino_5class/best.pt + the fine-tuned SAM3.
"""
import argparse, json, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
sys.path.insert(0, "/home/ps/sam3/sam3-inference")
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from product import load_sam3  # reuse SAM3(+ft head) loader

import rasterio
import rasterio.features
from rasterio.transform import from_bounds, Affine
from shapely.geometry import shape, mapping

DEV = "cuda"
NAMES = {0: "nodata", 1: "耕地", 2: "园地", 3: "林地", 4: "草地", 5: "其他"}
COLORS = {1: (0, 210, 0), 2: (170, 230, 0), 3: (0, 100, 0), 4: (130, 220, 140), 5: (160, 160, 160)}


@torch.no_grad()
def dino_5class(model, x6, cs=448):
    _, SZ, SZw = x6.shape
    acc = np.zeros((6, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
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
    return (acc / np.maximum(cnt, 1)).argmax(0)  # (H,W) 0-5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dino-ckpt", default="/mnt/sda/zf/landform/results/dino_5class/best.pt")
    p.add_argument("--sam3-weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--sam3-head", default="/mnt/sda/zf/landform/results/sam3_ft_fast/ft_eval.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--n-cells", type=int, default=4)
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--tile", type=int, default=740)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/product5")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModel
    dv = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    dino = DinoUNet5ch(dv, num_classes=6, in_channels=6, unfreeze_last_n=4).to(DEV)
    dino.load_state_dict(torch.load(a.dino_ckpt, map_location=DEV, weights_only=True)); dino.eval()
    proc = load_sam3(a.sam3_weights, a.sam3_head)
    print(f"[p5] models loaded ({time.time()-t0:.0f}s)", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    names = names[::max(1, len(names) // a.n_cells)][:a.n_cells]
    for nm in names:
        z = np.load(Path(a.data_dir) / f"{nm}.npz"); x6 = z["x6"]; bbox = z["bbox"]; H, W = x6.shape[1:]
        rgb = np.ascontiguousarray(x6[0:3].transpose(1, 2, 0))
        tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
        m5 = dino_5class(dino, x6)
        feats = []; counts = {c: 0 for c in range(1, 6)}
        wins = [(y, x, min(y + a.tile, H), min(x + a.tile, W)) for y in range(0, H, a.tile) for x in range(0, W, a.tile)]
        for (y0, x0, y1, x1) in wins:
            st = proc.set_image(Image.fromarray(rgb[y0:y1, x0:x1]))
            st = proc.set_text_prompt(prompt=a.prompt, state=st)
            mk = st.get("masks")
            if mk is None or mk.numel() == 0:
                continue
            wt = tr * Affine.translation(x0, y0)
            for inst in mk.squeeze(1).cpu().numpy().astype(np.uint8):
                if inst.sum() < 200:
                    continue
                sub = m5[y0:y1, x0:x1][inst > 0]; sub = sub[sub > 0]
                if sub.size == 0:
                    continue
                cls = int(np.bincount(sub, minlength=6).argmax()); counts[cls] = counts.get(cls, 0) + 1
                conf = float((sub == cls).mean())
                for geom, _ in rasterio.features.shapes(inst, mask=inst > 0, transform=wt):
                    poly = shape(geom).simplify(2 * abs(tr.a), preserve_topology=True)
                    if poly.area > 0:
                        feats.append({"type": "Feature", "geometry": mapping(poly),
                                      "properties": {"landform": NAMES[cls], "class_id": cls, "conf": round(conf, 3)}})
        (out / f"{nm}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))
        with rasterio.open(out / f"{nm}_landform.tif", "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="uint8", crs="EPSG:4326", transform=tr) as ds:
            ds.write(m5.astype(np.uint8), 1)
        s = max(1, H // 900); im = Image.fromarray(rgb[::s, ::s]); dr = ImageDraw.Draw(im)
        for f in feats:
            col = COLORS[f["properties"]["class_id"]]; geom = shape(f["geometry"])
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for pp in polys:
                xs, ys = pp.exterior.xy
                dr.line([((~tr * (x, y))[0] / s, (~tr * (x, y))[1] / s) for x, y in zip(xs, ys)], fill=col, width=2)
        im.save(out / f"{nm}_5cls.png")
        cs = " ".join(f"{NAMES[c]}{counts[c]}" for c in range(1, 6) if counts.get(c))
        print(f"  {nm}: {len(feats)} parcels [{cs}] -> geojson+tif+png", flush=True)
    print(f"[p5] done -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
