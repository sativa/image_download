"""TTA (test-time augmentation) evaluation for v14's best.pt.

Runs the trained model on the 8 balanced test cells with 4× augmentation:
  - identity
  - horizontal flip
  - vertical flip
  - h+v flip (180° rotation)

For each augmentation, predictions are flipped back to the original
orientation before averaging softmax probabilities. The final per-pixel
class is the argmax of the averaged probabilities.

Compares against the baseline (single-pass) IoU recorded during training.
Expected gain: +0.01-0.02 macro IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))

from train_v12_unet import (
    DEFAULT_DINOV2, DLTB_CLASS_TO_ID, ID_TO_DLTB, IMAGENET_MEAN, IMAGENET_STD,
    DLBM_TO_CLASS, rasterise_dltb_region, DinoUNet,
)


def slide_predict_softmax(model, rgb, device, stride=384, batch_size=4, flip=None):
    """Slide UNet over image; return per-pixel SOFTMAX probabilities."""
    if flip == "h":
        rgb = rgb[:, ::-1, :]
    elif flip == "v":
        rgb = rgb[::-1, :, :]
    elif flip == "hv":
        rgb = rgb[::-1, ::-1, :]
    rgb = np.ascontiguousarray(rgb)

    H, W = rgb.shape[:2]
    TILE = 448
    pad_h = (stride - (H - TILE) % stride) % stride if H > TILE else TILE - H
    pad_w = (stride - (W - TILE) % stride) % stride if W > TILE else TILE - W
    pad_h = max(0, pad_h); pad_w = max(0, pad_w)
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    Hp, Wp = padded.shape[:2]

    score = np.zeros((Hp, Wp, 6), dtype=np.float32)
    weight = np.zeros((Hp, Wp), dtype=np.float32)
    tiles, positions = [], []
    for top in range(0, Hp - TILE + 1, stride):
        for left in range(0, Wp - TILE + 1, stride):
            tiles.append(padded[top:top+TILE, left:left+TILE])
            positions.append((top, left))
    with torch.no_grad():
        for b0 in range(0, len(tiles), batch_size):
            batch = tiles[b0:b0+batch_size]
            arr = np.stack(batch).astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, (top, left) in enumerate(positions[b0:b0+len(batch)]):
                score[top:top+TILE, left:left+TILE] += probs[k].transpose(1, 2, 0)
                weight[top:top+TILE, left:left+TILE] += 1.0
    score /= np.maximum(weight, 1e-6)[..., None]
    score = score[:H, :W]
    # Un-flip
    if flip == "h":
        score = score[:, ::-1, :]
    elif flip == "v":
        score = score[::-1, :, :]
    elif flip == "hv":
        score = score[::-1, ::-1, :]
    return np.ascontiguousarray(score)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=HOME / "results/v14/best.pt")
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v11_regions_balanced.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--flips", default="identity,h,v,hv",
                   help="comma-separated subset of {identity,h,v,hv}")
    args = p.parse_args()

    flips_list = [f.strip() for f in args.flips.split(",")]
    print(f"TTA flips: {flips_list}", flush=True)
    device = args.device

    print(f"\n[1] Loading test data + model", flush=True)
    regions_meta = json.loads(args.regions_json.read_text())
    import geopandas as gpd
    import rasterio
    gdf_per_county = {}
    for r in regions_meta["test"]:
        code = r["county"]
        if code in gdf_per_county:
            continue
        g = gpd.read_parquet(args.dltb_cache / f"{code}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g

    test = []
    for r in regions_meta["test"]:
        bb = tuple(r["bbox"])
        for src in ["esri", "google"]:
            path = args.data_cache / f"{r['county']}_{r['idx']}_{src}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as rs:
                bands = rs.read(out_dtype="uint8")
                transform = rs.transform; H, W = rs.height, rs.width
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            lbl = rasterise_dltb_region(gdf_per_county[r["county"]], bb, transform, H, W)
            test.append((f"{r['county']}_{r['idx']}_{src}", rgb, lbl))
    print(f"  {len(test)} test images", flush=True)

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=6, unfreeze_last_n=4).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = getattr(model, "eval")()
    print(f"  loaded {args.checkpoint}", flush=True)

    print(f"\n[2] Running TTA inference + evaluation", flush=True)
    all_acc, all_macro = [], []
    per_class = {c: [] for c in range(1, 6)}
    t0 = time.time()
    for name, rgb, lbl in test:
        # Average softmax across flips
        avg_probs = np.zeros((rgb.shape[0], rgb.shape[1], 6), dtype=np.float32)
        for flip in flips_list:
            flip_arg = None if flip == "identity" else flip
            probs = slide_predict_softmax(model, rgb, device, flip=flip_arg)
            avg_probs += probs
        avg_probs /= len(flips_list)
        pred = avg_probs.argmax(axis=-1).astype(np.uint8)
        valid = lbl > 0
        if not valid.any():
            print(f"  {name}: no labelled pixels"); continue
        p, t = pred[valid], lbl[valid]
        acc = float((p == t).mean())
        classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
        classes = [c for c in classes if c != 0]
        ious = {}
        for c in classes:
            inter = int(((p == c) & (t == c)).sum())
            union = int(((p == c) | (t == c)).sum())
            ious[int(c)] = inter / union if union else 0.0
        macro = float(np.mean(list(ious.values()))) if ious else 0.0
        all_acc.append(acc); all_macro.append(macro)
        for c, v in ious.items():
            per_class.setdefault(c, []).append(v)
        print(f"  {name}: acc={acc:.3f} iou={macro:.3f}", flush=True)

    avg_acc = float(np.mean(all_acc))
    avg_macro = float(np.mean(all_macro))
    per_cls_str = " ".join(
        f"{ID_TO_DLTB.get(c, str(c))}:{np.mean(v):.3f}"
        for c, v in per_class.items() if v
    )
    print(f"\n[done] {time.time()-t0:.0f}s")
    print(f"  AVG: acc={avg_acc:.3f}  macro_iou={avg_macro:.3f}")
    print(f"  per-class: {per_cls_str}")
    print(f"\n  v14 baseline (no TTA): 0.201 macro_iou")
    print(f"  v14 + TTA {flips_list}: {avg_macro:.3f}")
    print(f"  delta: {avg_macro - 0.201:+.3f}")


if __name__ == "__main__":
    main()
