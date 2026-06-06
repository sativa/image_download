"""v8 ported for the remote 4× RTX 4090 box.

Paths point to ~/landform/* on the remote. Picks GPU 0 by default; pass
--device cuda:1 etc to spread across GPUs. Tile-based DINOv2 features
with stride 192 over 224 tiles. Uses CUDA, batches 32 tiles per step.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


HOME = Path("/home/ps/landform")
DLTB_PATH = HOME / "data/合水县_DLTB_classified.geoparquet"
DEFAULT_DINOV2 = HOME / "dinov2/dinov2-large"

DLTB_CLASS_TO_ID = {"耕地": 1, "园地": 2, "林地": 3, "草地": 4, "其他": 5}
ID_TO_DLTB = {v: k for k, v in DLTB_CLASS_TO_ID.items()}

TRAIN_BBOXES = [
    (107.9831, 35.7923, 108.0031, 35.8123),
    (107.9031, 35.8523, 107.9231, 35.8723),
    (108.0431, 35.6923, 108.0631, 35.7123),
    (108.1031, 35.8523, 108.1231, 35.8723),
    (107.92, 35.79, 107.94, 35.81),
    (108.00, 35.85, 108.02, 35.87),
    (108.04, 35.78, 108.06, 35.80),
    (108.06, 35.85, 108.08, 35.87),
    (107.99, 35.74, 108.01, 35.76),
    (108.12, 35.79, 108.14, 35.81),
    (108.04, 35.92, 108.06, 35.94),
    (107.88, 35.92, 107.90, 35.94),
]
TEST_BBOX = (107.8631, 35.7523, 107.8831, 35.7723)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TiledDinoExtractor:
    def __init__(self, weights_dir: str, device: str = "cuda:0",
                 tile: int = 224, stride: int = 192, batch_size: int = 32):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(weights_dir).to(device)
        self.model = getattr(self.model, "eval")()
        self.device = device
        self.tile = tile
        self.stride = stride
        self.patch = 14
        self.tile_patches = tile // self.patch
        self.batch_size = batch_size

    def __call__(self, rgb: np.ndarray):
        H, W = rgb.shape[:2]
        pad_h = (self.stride - (H - self.tile) % self.stride) % self.stride if H > self.tile else self.tile - H
        pad_w = (self.stride - (W - self.tile) % self.stride) % self.stride if W > self.tile else self.tile - W
        pad_h = max(0, pad_h); pad_w = max(0, pad_w)
        padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        Hp, Wp = padded.shape[:2]
        tiles, positions = [], []
        for top in range(0, Hp - self.tile + 1, self.stride):
            for left in range(0, Wp - self.tile + 1, self.stride):
                tiles.append(padded[top:top + self.tile, left:left + self.tile, :])
                positions.append((top, left))
        out_Ph, out_Pw = Hp // self.patch, Wp // self.patch
        D = self.model.config.hidden_size
        feat = np.zeros((out_Ph, out_Pw, D), dtype=np.float32)
        weight = np.zeros((out_Ph, out_Pw), dtype=np.float32)
        with torch.no_grad():
            for b0 in range(0, len(tiles), self.batch_size):
                batch = tiles[b0:b0 + self.batch_size]
                arr = np.stack(batch).astype(np.float32) / 255.0
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
                out = self.model(pixel_values=x)
                tokens = out.last_hidden_state[:, 1:, :].reshape(
                    x.shape[0], self.tile_patches, self.tile_patches, D
                ).cpu().numpy()
                for k in range(x.shape[0]):
                    top, left = positions[b0 + k]
                    pi, pj = top // self.patch, left // self.patch
                    feat[pi:pi + self.tile_patches, pj:pj + self.tile_patches] += tokens[k]
                    weight[pi:pi + self.tile_patches, pj:pj + self.tile_patches] += 1.0
        weight = np.maximum(weight, 1e-6)
        feat /= weight[..., None]
        Ph_o = H // self.patch + (1 if H % self.patch else 0)
        Pw_o = W // self.patch + (1 if W % self.patch else 0)
        return feat[:Ph_o, :Pw_o], (Ph_o, Pw_o), (H, W)


def patches_with_labels(rgb, label_raster, extractor):
    feat, (Ph, Pw), (H, W) = extractor(rgb)
    D = feat.shape[-1]
    y_grid = np.zeros((Ph, Pw), dtype=np.int32)
    for i in range(Ph):
        y0 = int(i * H / Ph); y1 = int((i + 1) * H / Ph)
        for j in range(Pw):
            x0 = int(j * W / Pw); x1 = int((j + 1) * W / Pw)
            region = label_raster[y0:y1, x0:x1]
            ll = region[region > 0]
            if ll.size:
                vals, counts = np.unique(ll, return_counts=True)
                y_grid[i, j] = int(vals[counts.argmax()])
    X = feat.reshape(-1, D); y = y_grid.reshape(-1)
    keep = y > 0
    return X[keep], y[keep], (Ph, Pw), feat, y_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path, default=DLTB_PATH)
    p.add_argument("--weights", default=str(DEFAULT_DINOV2),
                   help="path to DINOv2 dir (large or giant)")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/train_v6",
                   help="folder containing region_*.tif from v6")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v8")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tile-batch", type=int, default=32)
    p.add_argument("--stride", type=int, default=192)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = ["esri", "google"]
    print(f"device: {args.device}  weights: {args.weights}")

    print(f"\n[1] Locating cached imagery in {args.data_cache}")
    train_jobs = []
    for i, bb in enumerate(TRAIN_BBOXES):
        for src in sources:
            p_ = args.data_cache / f"train_{i}_{src}.tif"
            if p_.exists():
                train_jobs.append((bb, src, p_))
    test_path = args.data_cache / "test_esri.tif"
    if not test_path.exists():
        raise SystemExit(f"missing test image at {test_path}")
    print(f"  {len(train_jobs)} train images, test image ok")

    print(f"\n[2] Rasterising DLTB")
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize as _rasterize
    from shapely.geometry import box as shp_box
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try: full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError: full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)
    print(f"  loaded {len(full_g)} polygons")

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

    print(f"\n[3] Tile-based DINOv2 feature extraction on {args.device}")
    extractor = TiledDinoExtractor(args.weights, device=args.device,
                                    batch_size=args.tile_batch, stride=args.stride)
    train_X, train_y = [], []
    test_data = None
    t0 = time.time()
    for k, (bb, src, path) in enumerate(train_jobs):
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
            transform = rs.transform; H, W = rs.height, rs.width
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        lbl = _label(bb, transform, H, W)
        X, y, (Ph, Pw), feat, y_grid = patches_with_labels(rgb, lbl, extractor)
        train_X.append(X); train_y.append(y)
        if k % 6 == 0 or k == len(train_jobs) - 1:
            print(f"  train {k+1}/{len(train_jobs)}: {Ph}×{Pw} patches  ({time.time()-t0:.0f}s)")

    with rasterio.open(test_path) as rs:
        bands = rs.read(out_dtype="uint8")
        transform = rs.transform; H_test, W_test = rs.height, rs.width
    test_rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
    test_lbl = _label(TEST_BBOX, transform, H_test, W_test)
    test_X, test_y, (Pht, Pwt), test_feat, _ = patches_with_labels(test_rgb, test_lbl, extractor)
    print(f"  test: {Pht}×{Pwt} patches ({time.time()-t0:.0f}s total)")

    train_X = np.concatenate(train_X); train_y = np.concatenate(train_y)
    print(f"\n  train patches: {len(train_X):,}  class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    print(f"\n[4] Training MLP head")
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

    print(f"\n[5] Evaluation")
    D = test_feat.shape[-1]
    pred_grid = clf.predict(scaler.transform(test_feat.reshape(-1, D))).reshape(Pht, Pwt)
    pred_full = np.zeros((H_test, W_test), dtype=np.uint8)
    for i in range(Pht):
        y0, y1 = int(i*H_test/Pht), int((i+1)*H_test/Pht)
        for j in range(Pwt):
            x0, x1 = int(j*W_test/Pwt), int((j+1)*W_test/Pwt)
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
    print(f"  test acc: {acc:.3f}  macro IoU: {macro:.3f}")
    for c in classes:
        print(f"    {ID_TO_DLTB.get(c, str(c)):<6}: {ious[c]:.3f}")
    print()
    print(f"  v3 (32×32 patches @ 448): 38.3% / 0.238")
    print(f"  v7 fine-tune            : 40.8% / 0.203")
    print(f"  v8 tiled ({Pht}×{Pwt}, {args.weights.split('/')[-1]}): {acc:.1%} / {macro:.3f}")


if __name__ == "__main__":
    main()
