"""Load v9 best.pt and render input | DLTB truth | prediction side-by-side."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v9_finetune_gpu import (
    DEFAULT_DINOV2, DEFAULT_DLTB, V6_CACHE,
    DLTB_CLASS_TO_ID, ID_TO_DLTB, TEST_BBOX,
    _rasterise_label, evaluate_full_image, DinoWithHead, IMAGENET_MEAN, IMAGENET_STD,
)


PALETTE = {
    0: (0, 0, 0),
    1: (218, 165, 32),    # 耕地 (cropland) — golden
    2: (255, 105, 180),   # 园地 (orchard)  — pink
    3: (34, 139, 34),     # 林地 (forest)   — forest green
    4: (124, 252, 0),     # 草地 (grass)    — lime green
    5: (128, 128, 128),   # 其他 (other)    — gray
}


def main():
    device = "cuda:0"
    weights = HOME / "results/v9/best.pt"
    print(f"loading v9 best.pt")
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoWithHead(dinov2, num_classes=6, unfreeze_last_n=4).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = getattr(model, "eval")()
    print(f"  loaded")

    import rasterio, geopandas as gpd
    full_g = gpd.read_parquet(DEFAULT_DLTB).to_crs("EPSG:4326")
    try: full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError: full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)

    test_path = V6_CACHE / "test_esri.tif"
    with rasterio.open(test_path) as rs:
        bands = rs.read(out_dtype="uint8")
        transform = rs.transform; H, W = rs.height, rs.width
    rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
    truth = _rasterise_label(full_g, TEST_BBOX, transform, H, W)

    print(f"running inference on {H}x{W}")
    acc, macro, ious, pred_full = evaluate_full_image(
        model, rgb, truth, device, stride=192, batch_size=16,
    )
    print(f"acc={acc:.3f} macro_iou={macro:.3f}")
    for c in sorted(ious):
        print(f"  {ID_TO_DLTB.get(c, str(c)):<4}: {ious[c]:.3f}")

    # Render 3-panel side-by-side image.
    from PIL import Image, ImageDraw
    def colorize(lbl):
        h, w = lbl.shape
        rgba = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, col in PALETTE.items():
            rgba[lbl == cid] = col
        return (rgb.astype(np.float32) * 0.5 + rgba.astype(np.float32) * 0.5).astype(np.uint8)

    a = rgb
    b = colorize(truth)
    c = colorize(pred_full)
    LH = 30
    canvas = Image.new("RGB", (W*3, H + LH), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (lbl, arr) in enumerate([("input", a), ("DLTB truth", b), ("v9 prediction", c)]):
        draw.text((i*W + 10, 5), lbl, fill="black")
        canvas.paste(Image.fromarray(arr), (i*W, LH))
    canvas.thumbnail((2400, 1400))
    out = HOME / "results/v9/visual.png"
    canvas.save(out, optimize=True)
    print(f"saved {out} ({canvas.size})")


if __name__ == "__main__":
    main()
