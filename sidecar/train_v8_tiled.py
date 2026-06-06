"""v8: tile-based DINOv2 inference for fine-grained patch features.

Instead of feeding the whole region (e.g. 2560×2048) into DINOv2 with
positional-encoding interpolation (which v3 showed degrades quality
past 448), we slice each region into overlapping 224×224 tiles and
run DINOv2 at its NATIVE resolution on each tile. The patches across
tiles concatenate into a much finer effective grid:

  - v3 @ 448px : 32×32 patch grid → each patch ≈ 80×64 original px
  - v8 tiled   : up to 96×96 → each patch ≈ 27×21 original px (≈3× finer)

Stride 192 (overlap 32) avoids boundary artefacts where DINOv2's
attention drops off near tile edges. Overlapping patches at the seam
are averaged.
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


sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_supervised import DLTB_CLASS_TO_ID, ID_TO_DLTB
from train_v2 import TRAIN_BBOXES, TEST_BBOX
from train_v6_multisource import download_region_source


DINOV2_DIR = "/Users/zhangfeng/D/dinov2_weights/dinov2-large"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TiledDinoExtractor:
    """DINOv2 features at native 224×224, applied as overlapping tiles.

    Each call:
      1. Pads the image so (H, W) are multiples of stride.
      2. Slides a 224×224 window with stride 192 (32-px overlap).
      3. Batches tiles through DINOv2 (16 per batch on CPU, more on GPU).
      4. Each tile produces 16×16 = 256 patches → places them at the
         correct (i,j) of the output feature canvas.
      5. Overlapping patches are mean-pooled.

    Returns (feat_canvas[Ph, Pw, D], (Ph, Pw), (H, W)).
    """

    def __init__(self, device: str = "cpu", tile: int = 224, stride: int = 192,
                 batch_size: int = 8):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(DINOV2_DIR).to(device)
        self.model = getattr(self.model, "eval")()
        self.device = device
        self.tile = tile
        self.stride = stride
        self.patch = 14  # DINOv2's patch size
        # 16 patches per tile side (224/14), so per-tile patch grid is 16×16.
        self.tile_patches = tile // self.patch
        self.batch_size = batch_size

    def __call__(self, rgb: np.ndarray):
        from PIL import Image
        H, W = rgb.shape[:2]
        # Pad to multiples of stride so the last tile fits.
        pad_h = (self.stride - (H - self.tile) % self.stride) % self.stride if H > self.tile else self.tile - H
        pad_w = (self.stride - (W - self.tile) % self.stride) % self.stride if W > self.tile else self.tile - W
        pad_h = max(0, pad_h); pad_w = max(0, pad_w)
        padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        Hp, Wp = padded.shape[:2]

        # Patch grid for the full padded image. Since stride / patch=192/14
        # isn't integer, we instead anchor tiles on a grid and stamp their
        # patch outputs into a per-tile (16×16) cell, snapping to the
        # nearest output patch coords. To keep this simple and correct, we
        # build the output canvas at "tile-native" patch resolution by
        # iterating tile positions in patch units.
        tile_step_px = self.stride
        tiles = []
        positions = []  # (top_pix, left_pix) in PADDED image coords
        for top in range(0, Hp - self.tile + 1, tile_step_px):
            for left in range(0, Wp - self.tile + 1, tile_step_px):
                tiles.append(padded[top:top + self.tile, left:left + self.tile, :])
                positions.append((top, left))
        # Output canvas: one patch per `patch` (14) px in the padded image.
        out_Ph = Hp // self.patch
        out_Pw = Wp // self.patch
        D = self.model.config.hidden_size
        feat_canvas = np.zeros((out_Ph, out_Pw, D), dtype=np.float32)
        weight_canvas = np.zeros((out_Ph, out_Pw), dtype=np.float32)

        # Batch tiles through the model.
        with torch.no_grad():
            for b0 in range(0, len(tiles), self.batch_size):
                batch_imgs = tiles[b0:b0 + self.batch_size]
                arr = np.stack(batch_imgs).astype(np.float32) / 255.0
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
                out = self.model(pixel_values=x)
                tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
                Bsz = tokens.shape[0]
                feat_tiles = tokens.reshape(Bsz, self.tile_patches, self.tile_patches, D)
                feat_tiles = feat_tiles.cpu().numpy()
                for k in range(Bsz):
                    top, left = positions[b0 + k]
                    pi = top // self.patch
                    pj = left // self.patch
                    feat_canvas[pi:pi + self.tile_patches, pj:pj + self.tile_patches] += feat_tiles[k]
                    weight_canvas[pi:pi + self.tile_patches, pj:pj + self.tile_patches] += 1.0

        # Average overlapping patches; zero-weight cells (shouldn't happen)
        # get a tiny eps so we don't divide by zero.
        weight_canvas = np.maximum(weight_canvas, 1e-6)
        feat_canvas /= weight_canvas[..., None]

        # Crop back to (H, W) in patch units.
        out_Ph_orig = H // self.patch + (1 if H % self.patch else 0)
        out_Pw_orig = W // self.patch + (1 if W % self.patch else 0)
        feat_canvas = feat_canvas[:out_Ph_orig, :out_Pw_orig]
        return feat_canvas, (out_Ph_orig, out_Pw_orig), (H, W)


def extract_patch_samples_tiled(rgb, label_raster, extractor):
    feat, (Ph, Pw), (H, W) = extractor(rgb)
    D = feat.shape[-1]
    y_grid = np.zeros((Ph, Pw), dtype=np.int32)
    patch_size = H // Ph + (1 if H % Ph else 0)  # ≈14
    # Use H/Ph mapping for per-patch label.
    for i in range(Ph):
        y0 = int(i * H / Ph); y1 = int((i + 1) * H / Ph)
        for j in range(Pw):
            x0 = int(j * W / Pw); x1 = int((j + 1) * W / Pw)
            region = label_raster[y0:y1, x0:x1]
            labelled = region[region > 0]
            if labelled.size == 0:
                continue
            vals, counts = np.unique(labelled, return_counts=True)
            y_grid[i, j] = int(vals[counts.argmax()])
    X_flat = feat.reshape(-1, D)
    y_flat = y_grid.reshape(-1)
    keep = y_flat > 0
    return X_flat[keep], y_flat[keep], (Ph, Pw), feat, y_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v8"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--sources", default="esri,google")
    p.add_argument("--tile-batch", type=int, default=8)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = args.sources.split(",")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # 1) Gather imagery — reuse v6's cache.
    print(f"[1] Gathering imagery (reuse v6 cache)")
    session = requests.Session(); session.headers["User-Agent"] = "trainv8/1.0"
    train_jobs = []
    for i, bb in enumerate(TRAIN_BBOXES):
        for src in sources:
            cache = Path("/tmp/train_v6") / f"train_{i}_{src}.tif"
            tgt = cache if cache.exists() else (args.out_dir / f"train_{i}_{src}.tif")
            train_jobs.append((bb, src, tgt))
    test_cache = Path("/tmp/train_v6/test_esri.tif")
    test_path = test_cache if test_cache.exists() else (args.out_dir / "test_esri.tif")

    def _ensure(args_):
        bb, src, path = args_
        if path.exists():
            return bb, src, path
        download_region_source(bb, args.zoom, src, path, session)
        return bb, src, path

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(train_jobs) + 1, 32)) as ex:
        train_jobs = list(ex.map(_ensure, train_jobs))
    if not test_path.exists():
        download_region_source(TEST_BBOX, args.zoom, "esri", test_path, session)
    print(f"  ready in {time.time()-t0:.1f}s")

    # 2) Rasterise DLTB
    print(f"\n[2] Rasterising DLTB")
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize as _rasterize
    from shapely.geometry import box as shp_box
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try: full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError: full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)

    label_cache: dict = {}
    def _label(bb, transform, H, W):
        k = (bb, H, W)
        if k in label_cache: return label_cache[k]
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

    # 3) Tile-extract features for every image
    print(f"\n[3] Tiled DINOv2 feature extraction")
    extractor = TiledDinoExtractor(device=device, batch_size=args.tile_batch)
    train_X, train_y = [], []
    test_data = None
    t0 = time.time()
    for k, (bb, src, path) in enumerate(train_jobs):
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
            transform = rs.transform; H, W = rs.height, rs.width
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        lbl = _label(bb, transform, H, W)
        X, y, (Ph, Pw), feat, y_grid = extract_patch_samples_tiled(rgb, lbl, extractor)
        train_X.append(X); train_y.append(y)
        if k % 6 == 0:
            print(f"  train {k+1}/{len(train_jobs)}: {Ph}×{Pw} patches, {len(X)} labelled "
                  f"({time.time()-t0:.0f}s elapsed)")

    with rasterio.open(test_path) as rs:
        bands = rs.read(out_dtype="uint8")
        transform = rs.transform; H_test, W_test = rs.height, rs.width
    test_rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
    test_lbl = _label(TEST_BBOX, transform, H_test, W_test)
    test_X, test_y, (Pht, Pwt), test_feat, test_y_grid = extract_patch_samples_tiled(test_rgb, test_lbl, extractor)
    print(f"  test image: {Pht}×{Pwt} patches, {len(test_X)} labelled")

    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"\n  TOTAL train patches: {len(train_X)} ({time.time()-t0:.0f}s)")
    print(f"  class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    # 4) Train MLP
    print(f"\n[4] Training MLP")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_X)
    Xs = scaler.transform(train_X)
    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                        early_stopping=True, random_state=0)
    clf.fit(Xs, train_y)
    print(f"  trained in {time.time()-t0:.1f}s, train_acc={clf.score(Xs, train_y):.3f}")

    import joblib
    joblib.dump({"clf": clf, "scaler": scaler, "id_to_label": ID_TO_DLTB},
                args.out_dir / "head_v8.joblib")

    # 5) Evaluate
    print(f"\n[5] Evaluation")
    D = test_feat.shape[-1]
    pred_grid = clf.predict(scaler.transform(test_feat.reshape(-1, D))).reshape(Pht, Pwt)
    pred_full = np.zeros((H_test, W_test), dtype=np.uint8)
    for i in range(Pht):
        y0 = int(i * H_test / Pht); y1 = int((i + 1) * H_test / Pht)
        for j in range(Pwt):
            x0 = int(j * W_test / Pwt); x1 = int((j + 1) * W_test / Pwt)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = test_lbl > 0
    p, t = pred_full[valid], test_lbl[valid]
    acc = float((p == t).mean())
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(ious.values()))) if ious else 0.0

    print(f"  test acc: {acc:.3f}")
    for c in classes:
        print(f"    {ID_TO_DLTB.get(c, str(c)):<6}: IoU={ious[c]:.3f}")
    print(f"  macro IoU: {macro:.3f}")
    print()
    print(f"  baseline             : 26.4% / 0.107")
    print(f"  v3 @ 448 (32×32)     : 38.3% / 0.238")
    print(f"  v6 multi-src (32×32) : 37.8% / 0.225")
    print(f"  v7 fine-tune (32×32) : 40.8% / 0.203")
    print(f"  v8 tiled ({Pht}×{Pwt})    : {acc:.1%} / {macro:.3f}")


if __name__ == "__main__":
    main()
