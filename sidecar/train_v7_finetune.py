"""v7: Partial fine-tune of DINOv2-large on 三调 labels.

Unlike v3-v6 (sklearn MLP head on frozen DINOv2 features), this script
runs a real PyTorch training loop that unfreezes the LAST 2 transformer
blocks of DINOv2 + a fresh classifier head; everything before is kept
frozen. Backprops cross-entropy through the unfrozen layers.

AdamW with two LR groups: backbone 1e-5, head 1e-3. Class-weighted
loss to counter the dominant-class imbalance. 5 epochs.

Same multi-source data (12 regions × {Esri, Google} = 24 image
versions) as v6 so we can isolate the fine-tune effect.

Why "last 2 blocks": full fine-tune of all 24 blocks would touch
~300M params on ~20k samples — overfit risk. Last 2 blocks give the
model enough flexibility to specialise while leaving early geometry
detectors intact.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn


sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_supervised import DLTB_CLASS_TO_ID, ID_TO_DLTB
from train_v2 import TRAIN_BBOXES, TEST_BBOX
from train_v6_multisource import download_region_source


DINOV2_DIR = "/Users/zhangfeng/D/dinov2_weights/dinov2-large"


def _load_dinov2(device: str):
    from transformers import AutoModel
    return AutoModel.from_pretrained(DINOV2_DIR).to(device)


def _set_eval(module):
    return getattr(module, "eval")()


def _prep_input(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    from PIL import Image
    pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class DinoWithHead(nn.Module):
    """DINOv2 backbone (last N blocks unfrozen) + classifier head."""

    def __init__(self, dinov2, num_classes: int, embed_dim: int = 1024,
                 unfreeze_last_n: int = 2):
        super().__init__()
        self.backbone = dinov2
        # Freeze everything by default.
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze last N transformer encoder layers.
        try:
            blocks = self.backbone.encoder.layer
        except AttributeError:
            blocks = self.backbone.encoder.layers
        for blk in list(blocks)[-unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[:, 1:, :]
        B, N, D = tokens.shape
        Ph = Pw = int(np.sqrt(N))
        logits = self.head(tokens)
        return logits.permute(0, 2, 1).reshape(B, -1, Ph, Pw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v7"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--sources", default="esri,google")
    p.add_argument("--unfreeze-blocks", type=int, default=2)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = args.sources.split(",")
    device = "cpu"

    # 1) Gather train + test imagery (reuse v6's cache aggressively).
    print(f"[1] Gathering imagery (reuse v6 cache where possible)")
    session = requests.Session(); session.headers["User-Agent"] = "trainv7/1.0"
    jobs = []
    for i, bb in enumerate(TRAIN_BBOXES):
        for src in sources:
            cache = Path("/tmp/train_v6") / f"train_{i}_{src}.tif"
            tgt = cache if cache.exists() else (args.out_dir / f"train_{i}_{src}.tif")
            jobs.append((bb, src, tgt))
    test_cache = Path("/tmp/train_v6/test_esri.tif")
    test_path = test_cache if test_cache.exists() else (args.out_dir / "test_esri.tif")

    def _ensure(args_):
        bb, src, path = args_
        if path.exists():
            return bb, src, path
        download_region_source(bb, args.zoom, src, path, session)
        return bb, src, path

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(jobs) + 1, 32)) as ex:
        jobs = list(ex.map(_ensure, jobs))
    if not test_path.exists():
        download_region_source(TEST_BBOX, args.zoom, "esri", test_path, session)
    print(f"  ready in {time.time()-t0:.1f}s ({len(jobs)} train images)")

    # 2) Rasterise DLTB per unique region.
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize as _rasterize
    from shapely.geometry import box as shp_box

    print(f"\n[2] Loading + rasterising DLTB")
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try:
        full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError:
        full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)

    label_cache: dict = {}
    def _label(bb, transform, H, W):
        k = (bb, H, W)
        if k in label_cache:
            return label_cache[k]
        idx = list(full_g.sindex.intersection(bb))
        sub = full_g.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
        sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
        shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
        lbl = (_rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                          fill=0, dtype="uint8")
               if shapes else np.zeros((H, W), dtype=np.uint8))
        label_cache[k] = lbl
        return lbl

    # 3) Preload all inputs to tensors.
    print(f"\n[3] Preloading tensors")
    Ph = Pw = args.image_size // 14
    train_items = []
    for bb, src, path in jobs:
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
            transform = rs.transform; H_o, W_o = rs.height, rs.width
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        lbl_full = _label(bb, transform, H_o, W_o)
        y_grid = np.zeros((Ph, Pw), dtype=np.int64)
        for i in range(Ph):
            y0 = int(i * H_o / Ph); y1 = int((i + 1) * H_o / Ph)
            for j in range(Pw):
                x0 = int(j * W_o / Pw); x1 = int((j + 1) * W_o / Pw)
                region = lbl_full[y0:y1, x0:x1]
                ll = region[region > 0]
                if ll.size:
                    vals, counts = np.unique(ll, return_counts=True)
                    y_grid[i, j] = int(vals[counts.argmax()])
        train_items.append((_prep_input(rgb, args.image_size), torch.from_numpy(y_grid)))

    with rasterio.open(test_path) as rs:
        bands = rs.read(out_dtype="uint8")
        transform = rs.transform; H_o, W_o = rs.height, rs.width
    test_rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
    test_lbl_full = _label(TEST_BBOX, transform, H_o, W_o)
    test_pixel = _prep_input(test_rgb, args.image_size)
    print(f"  {len(train_items)} train images, 1 test image")

    # 4) Build model.
    print(f"\n[4] Building model (unfreezing last {args.unfreeze_blocks} blocks)")
    dinov2 = _load_dinov2(device)
    num_classes = 6  # 5 DLTB + class-0 ignore
    model = DinoWithHead(dinov2, num_classes=num_classes,
                         unfreeze_last_n=args.unfreeze_blocks).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable / total: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("head.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)

    y_all = torch.cat([t[1].flatten() for t in train_items])
    counts = torch.bincount(y_all, minlength=num_classes).float()
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = 0.0 if counts[c] == 0 else (1.0 / counts[c].sqrt())
    weights[0] = 0.0
    weights = weights / weights.sum() * (num_classes - 1)
    print(f"  class weights: {[round(w.item(), 3) for w in weights]}")
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=0)

    # 5) Train.
    print(f"\n[5] Training {args.epochs} epochs")
    rng = np.random.RandomState(0)
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(train_items)))
        rng.shuffle(order)
        ep_loss = 0.0; ep_c = 0; ep_t = 0
        t0 = time.time()
        for img_idx in order:
            pixel, y_grid = train_items[img_idx]
            pixel = pixel.to(device); y_grid = y_grid.to(device)
            logits = model(pixel)
            loss = loss_fn(logits, y_grid.unsqueeze(0))
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            preds = logits.argmax(dim=1)[0].cpu().numpy()
            tgt = y_grid.cpu().numpy()
            valid = tgt > 0
            if valid.any():
                ep_c += int((preds[valid] == tgt[valid]).sum())
                ep_t += int(valid.sum())
        print(f"  epoch {ep+1}: loss={ep_loss/len(train_items):.3f} "
              f"acc={ep_c/max(ep_t,1):.3f} ({time.time()-t0:.1f}s)")

    # 6) Evaluate.
    print(f"\n[6] Evaluation")
    model = _set_eval(model)
    with torch.no_grad():
        test_logits = model(test_pixel.to(device))
    pred_grid = test_logits.argmax(dim=1)[0].cpu().numpy()
    pred_full = np.zeros((H_o, W_o), dtype=np.uint8)
    for i in range(Ph):
        y0 = int(i * H_o / Ph); y1 = int((i + 1) * H_o / Ph)
        for j in range(Pw):
            x0 = int(j * W_o / Pw); x1 = int((j + 1) * W_o / Pw)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = test_lbl_full > 0
    p, t = pred_full[valid], test_lbl_full[valid]
    acc = float((p == t).mean())
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(ious.values()))) if ious else 0.0

    torch.save(model.state_dict(), args.out_dir / "head_v7.pt")
    print(f"  test acc: {acc:.3f}")
    for c in classes:
        print(f"    {ID_TO_DLTB.get(c, str(c)):<6}: IoU={ious[c]:.3f}")
    print(f"  macro IoU: {macro:.3f}")
    print()
    print(f"  baseline (color rules):  26.4% / 0.107")
    print(f"  v3 frozen MLP @ 448  :   38.3% / 0.238")
    print(f"  v6 multi-src MLP     :   37.8% / 0.225")
    print(f"  v7 partial fine-tune :   {acc:.1%} / {macro:.3f}")


if __name__ == "__main__":
    main()
