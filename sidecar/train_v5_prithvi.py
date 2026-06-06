"""v5: Prithvi-EO-2.0-300M backend.

Adapts our 3-band Esri RGB imagery to Prithvi's expected 6-band input:
  - Map our R/G/B → Prithvi's B04 (red) / B03 (green) / B02 (blue)
  - Fill the missing 3 bands (B05 NIR, B06 SWIR1, B07 SWIR2) with each
    band's training-set mean, i.e. neutral input.
  - Scale 0-255 → ~0-10000 reflectance, then normalise (val - mean)/std.

Same 12-region training set + 5-class scheme as v3 for direct
comparability. Uses Prithvi's `forward_features` on its single-frame
mode (num_frames=1) at the model's default 224×224, producing a
14×14 patch grid per image. We resize the region image to 224×224
(same as v3 baseline at 224) for the fairest head-to-head.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests


sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_supervised import (
    DLTB_CLASS_TO_ID, ID_TO_DLTB, download_region, rasterise_dltb,
)
from train_v2 import TRAIN_BBOXES, TEST_BBOX
from train_v3 import extract_patch_samples


PRITHVI_DIR = Path("/Users/zhangfeng/D/prithvi_weights/Prithvi-EO-2.0-300M")
# Prithvi config — mean/std per Sentinel-2 band.
BAND_MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
BAND_STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)


def _import_prithvi_mae():
    """Load `prithvi_mae.py` from the weights dir as a module."""
    sys.path.insert(0, str(PRITHVI_DIR))
    spec = importlib.util.spec_from_file_location("prithvi_mae", PRITHVI_DIR / "prithvi_mae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrithviExtractor:
    """Feature extractor wrapping Prithvi-EO-2.0-300M.

    Crops or resizes the input to 224×224, maps RGB→6-band, runs the
    encoder, returns the (14, 14, embed_dim) patch grid.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        prithvi_mae = _import_prithvi_mae()
        # Recreate the model with the published config.
        config = {
            "img_size": 224, "num_frames": 1, "patch_size": (1, 16, 16),
            "in_chans": 6, "embed_dim": 1024, "depth": 24, "num_heads": 16,
            "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16,
            "mlp_ratio": 4.0, "norm_pix_loss": False, "coords_encoding": [],
            "coords_scale_learn": False, "mask_ratio": 0.0,
        }
        self.model = prithvi_mae.PrithviMAE(**config)
        ckpt = torch.load(PRITHVI_DIR / "Prithvi_EO_V2_300M.pt", map_location="cpu",
                          weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # The published checkpoint may carry the num_frames=4 weights;
        # strict=False lets the temporal positional buffers be re-init'd
        # for our num_frames=1 deployment.
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"  Prithvi load: missing={len(missing)} unexpected={len(unexpected)} keys")
        self.encoder = self.model.encoder.to(device)
        self.encoder = getattr(self.encoder, "eval")()
        self.device = device

    def __call__(self, rgb: np.ndarray):
        """rgb: (H, W, 3) uint8 — returns ((Ph, Pw, D), (Ph, Pw), (H, W))."""
        import torch
        from PIL import Image

        H, W = rgb.shape[:2]
        # Resize to model's expected 224×224.
        pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
        rgb224 = np.array(pil).astype(np.float32)  # (224, 224, 3)
        # Scale 0-255 → ~0-2500 (matches the surface-reflectance range
        # the model was normalised on). 2500/255 ≈ 9.8.
        rgb_scaled = rgb_to_reflectance(rgb224)  # (224, 224, 3) in S2 units
        # Build 6-channel: [B02, B03, B04, B05, B06, B07].
        # We have B (our blue), G (green), R (red); fill NIR/SWIRs with
        # their channel-mean.
        bands = np.zeros((6, 224, 224), dtype=np.float32)
        bands[0] = rgb_scaled[..., 2]  # B02 ← our B
        bands[1] = rgb_scaled[..., 1]  # B03 ← our G
        bands[2] = rgb_scaled[..., 0]  # B04 ← our R
        bands[3] = BAND_MEAN[3]
        bands[4] = BAND_MEAN[4]
        bands[5] = BAND_MEAN[5]
        # Apply Prithvi's per-band normalisation.
        bands = (bands - BAND_MEAN[:, None, None]) / BAND_STD[:, None, None]
        # (B, C, T=1, H, W) — Prithvi wants 5D inputs.
        x = torch.from_numpy(bands).unsqueeze(0).unsqueeze(2).to(self.device)
        with torch.no_grad():
            feats = self.encoder.forward_features(x)
        # Last block's output (excluding cls token) reshaped to (Ph, Pw, D).
        last = feats[-1][:, 1:, :]  # (1, n_tokens, D)
        n_tok, D = last.shape[1], last.shape[2]
        Ph = Pw = int(np.sqrt(n_tok))
        feat = last[0].detach().cpu().numpy().reshape(Ph, Pw, D)
        return feat, (Ph, Pw), (H, W)


def rgb_to_reflectance(rgb_float: np.ndarray) -> np.ndarray:
    """Stretch 0-255 uint8 RGB to approximate Sentinel-2 reflectance values.

    Sentinel-2 surface reflectance is typically reported in DN units
    ~0-10000 (×10000 = 100% reflectance). Our RGB jpegs have already
    been white-balanced/gamma-encoded for human viewing, so a linear
    rescale to a "reasonable" reflectance range is the most we can do
    without spectral information. 0-255 → 0-2500 puts most pixels in
    the same numeric range as Prithvi's per-band means (~1000-2700).
    """
    return rgb_float * (2500.0 / 255.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dltb", type=Path,
                   default=Path("/Volumes/ORICO/data_ana/landuse/合水县_DLTB_classified.geoparquet"))
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v5_prithvi"))
    p.add_argument("--zoom", type=int, default=17)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session(); session.headers["User-Agent"] = "trainv5/1.0"

    # Use exactly the same 12+1 region set as v3 for direct comparison.
    all_bboxes = TRAIN_BBOXES + [TEST_BBOX]
    paths = [args.out_dir / f"region_{i}.tif" for i in range(len(all_bboxes))]
    print(f"[1] Downloading {len(all_bboxes)} regions in parallel")

    def _dl(args_):
        i, bb = args_
        # Reuse v3 cached imagery if it exists.
        v3_cache = Path("/tmp/train_v3") / f"region_{i}.tif"
        if v3_cache.exists() and not paths[i].exists():
            import shutil; shutil.copy(v3_cache, paths[i])
        if paths[i].exists():
            import rasterio
            with rasterio.open(paths[i]) as src:
                return i, bb, paths[i], src.transform, src.height, src.width
        return (i, bb) + download_region(bb, args.zoom, paths[i], session)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(all_bboxes)) as ex:
        dl_results = list(ex.map(_dl, list(enumerate(all_bboxes))))
    print(f"  done in {time.time()-t0:.1f}s")

    # Rasterise DLTB — reuse train_v2's approach (load once, parallel clip).
    print(f"\n[2] Rasterising DLTB (5-class)")
    import geopandas as gpd
    from shapely.geometry import box as shp_box
    from rasterio.features import rasterize as _rasterize
    t0 = time.time()
    full_g = gpd.read_parquet(args.dltb).to_crs("EPSG:4326")
    try:
        full_g["geometry"] = full_g.geometry.make_valid()
    except AttributeError:
        full_g["geometry"] = full_g.geometry.buffer(0)
    full_g["cid"] = full_g["一级地类"].map(DLTB_CLASS_TO_ID).fillna(0).astype(int)
    print(f"  loaded {len(full_g)} polygons in {time.time()-t0:.1f}s")

    region_labels = {}
    def _ras(args_):
        i, bb, tif, transform, H, W = args_
        idx = list(full_g.sindex.intersection(bb))
        sub = full_g.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
        sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
        shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["cid"]) if c > 0]
        lbl = (_rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                          fill=0, dtype="uint8")
               if shapes else np.zeros((H, W), dtype=np.uint8))
        return i, (tif, transform, H, W, lbl)

    with ThreadPoolExecutor(max_workers=len(dl_results)) as ex:
        for i, payload in ex.map(_ras, dl_results):
            region_labels[i] = payload

    # Extract Prithvi features.
    print(f"\n[3] Loading + running Prithvi-EO-2.0-300M")
    extractor = PrithviExtractor()
    import rasterio
    train_X, train_y = [], []
    test_data = None
    t0 = time.time()
    for i in sorted(region_labels):
        tif, transform, H, W, lbl = region_labels[i]
        with rasterio.open(tif) as src:
            bands = src.read(out_dtype="uint8")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        X, y, (Ph, Pw), feat_grid, y_grid = extract_patch_samples(rgb, lbl, extractor)
        if i < len(TRAIN_BBOXES):
            train_X.append(X); train_y.append(y)
        else:
            test_data = (rgb, lbl, feat_grid, y_grid, Ph, Pw, H, W)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"  TOTAL train: {len(train_X)} patches in {time.time()-t0:.1f}s "
          f"(patch grid {Ph}x{Pw} per region)")
    print(f"  class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    # Train MLP — same architecture as v2/v3.
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

    # Evaluate.
    print(f"\n[5] Evaluation")
    rgb_test, lbl_test, feat_grid_test, y_grid_test, Ph, Pw, H, W = test_data
    D = feat_grid_test.shape[-1]
    pred_grid = clf.predict(scaler.transform(feat_grid_test.reshape(-1, D))).reshape(Ph, Pw)
    pred_full = np.zeros((H, W), dtype=np.uint8)
    for i in range(Ph):
        y0, y1 = int(i*H/Ph), int((i+1)*H/Ph)
        for j in range(Pw):
            x0, x1 = int(j*W/Pw), int((j+1)*W/Pw)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = lbl_test > 0
    p, t = pred_full[valid], lbl_test[valid]
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
    print(f"  per-class IoU:")
    for c in classes:
        print(f"    {ID_TO_DLTB.get(c, str(c)):<6}: {ious[c]:.3f}")
    print(f"  macro IoU: {macro:.3f}")

    print()
    print(f"  Comparison:")
    print(f"    color rules baseline : 26.4% / 0.107")
    print(f"    v3 DINOv2 @ 448px MLP: 38.3% / 0.238")
    print(f"    v5 Prithvi-300M MLP  : {acc:.1%} / {macro:.3f}")


if __name__ == "__main__":
    main()
