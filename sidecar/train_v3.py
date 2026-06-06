"""Iteration v3: higher-resolution DINOv2 features.

Forces DINOv2 to process the image at 448×448 instead of the
processor's default 224×224, producing a 32×32 = 1024 patch grid
(4× more samples per region). Each patch then covers ~80×64 px of
the original — finer-grained and more single-class.

Reuses v2's training pipeline (12 train + 1 test region, MLP head,
parallel-everything) but with the higher-res extractor.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests


sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_supervised import (
    DLTB_CLASS_TO_ID, ID_TO_DLTB, download_region,
)
from train_v2 import TRAIN_BBOXES, TEST_BBOX


class DinoExtractorHiRes:
    """DINOv2 with `interpolate_pos_encoding=True` and bigger input.

    DINOv2's positional encodings are learned at the model's native
    resolution (16×16 patch grid = 224×224 input for base/large). When
    you feed a larger input the model can interpolate its positional
    encodings on the fly via the `interpolate_pos_encoding` flag — and
    the patch grid scales proportionally.
    """

    def __init__(self, image_size: int = 448, device: str = "cpu",
                 weights_dir: str = "/Users/zhangfeng/D/dinov2_weights/dinov2-large"):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(weights_dir)
        # Override the processor's target size — it normally crops/resizes
        # to 224. The model's interpolate_pos_encoding handles the rest.
        if hasattr(self.processor, "size"):
            self.processor.size = {"height": image_size, "width": image_size}
        if hasattr(self.processor, "do_resize"):
            self.processor.do_resize = True
        self.model = AutoModel.from_pretrained(weights_dir).to(device)
        self.model = getattr(self.model, "eval")()
        self.device = device
        self.image_size = image_size

    def __call__(self, rgb: np.ndarray):
        """Manually pre-resize to image_size, then ImageNet-normalise.

        The HF AutoImageProcessor has its own crop_size that quietly
        overrides our `size` kwarg, so we bypass its resize entirely
        and just hand-normalise the tensor — much more predictable.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        H, W = rgb.shape[:2]
        pil = Image.fromarray(rgb).resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        arr = np.array(pil).astype(np.float32) / 255.0
        # ImageNet mean/std (DINOv2 was trained with these).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        pv = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=pv, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[0, 1:, :]
        _, _, Hi, Wi = pv.shape
        Ph, Pw = Hi // 14, Wi // 14
        feat = tokens.detach().cpu().numpy().reshape(Ph, Pw, -1)
        return feat, (Ph, Pw), (H, W)


def extract_patch_samples(rgb, label_raster, dino_extractor):
    feat, (Ph, Pw), (H, W) = dino_extractor(rgb)
    D = feat.shape[-1]
    y_grid = np.zeros((Ph, Pw), dtype=np.int32)
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
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/train_v3"))
    p.add_argument("--image-size", type=int, default=448,
                   help="DINOv2 input size. 224=v2 default, 448=4× finer, 672=9× finer")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "trainv3/1.0"

    all_bboxes = TRAIN_BBOXES + [TEST_BBOX]
    paths = [args.out_dir / f"region_{i}.tif" for i in range(len(all_bboxes))]
    # Reuse v2's cached imagery if same out-dir, otherwise download.
    print(f"[1] Downloading {len(all_bboxes)} regions in parallel")

    def _dl(args_):
        i, bb = args_
        # Reuse v2 cached imagery if it exists.
        v2_path = Path("/tmp/train_v2") / f"region_{i}.tif"
        if v2_path.exists() and not paths[i].exists():
            import shutil
            shutil.copy(v2_path, paths[i])
        if paths[i].exists():
            import rasterio
            with rasterio.open(paths[i]) as src:
                return i, bb, paths[i], src.transform, src.height, src.width
        return (i, bb) + download_region(bb, args.zoom, paths[i], session)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(all_bboxes)) as ex:
        dl_results = list(ex.map(_dl, list(enumerate(all_bboxes))))
    print(f"  done in {time.time()-t0:.1f}s")

    # Rasterise DLTB once-loaded.
    print(f"\n[2] Rasterising DLTB")
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

    # Extract features at hi-res.
    print(f"\n[3] Extracting hi-res DINOv2 features (image_size={args.image_size})")
    dino = DinoExtractorHiRes(image_size=args.image_size)
    import rasterio
    train_X, train_y = [], []
    test_data = None
    t0 = time.time()
    for i in sorted(region_labels):
        tif, transform, H, W, lbl = region_labels[i]
        with rasterio.open(tif) as src:
            bands = src.read(out_dtype="uint8")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        X, y, (Ph, Pw), feat_grid, y_grid = extract_patch_samples(rgb, lbl, dino)
        if i < len(TRAIN_BBOXES):
            train_X.append(X); train_y.append(y)
        else:
            test_data = (rgb, lbl, feat_grid, y_grid, Ph, Pw, H, W)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    print(f"  TOTAL train: {len(train_X)} patches in {time.time()-t0:.1f}s")
    print(f"  patch grid: {Ph}×{Pw} per region")
    print(f"  train class hist: {dict(zip(*np.unique(train_y, return_counts=True)))}")

    # Train MLP only (winner from v2).
    print(f"\n[4] Training MLP head")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_X)
    Xs = scaler.transform(train_X)
    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                        early_stopping=True, random_state=0)
    clf.fit(Xs, train_y)
    print(f"  MLP trained in {time.time()-t0:.1f}s, train acc={clf.score(Xs, train_y):.3f}")

    import joblib
    joblib.dump({"clf": clf, "scaler": scaler, "image_size": args.image_size,
                 "id_to_label": ID_TO_DLTB},
                args.out_dir / "head_v3.joblib")

    # Evaluate.
    print(f"\n[5] Evaluation")
    rgb_test, lbl_test, feat_grid_test, y_grid_test, Ph, Pw, H, W = test_data
    D = feat_grid_test.shape[-1]
    Xall = feat_grid_test.reshape(-1, D)
    pred_grid = clf.predict(scaler.transform(Xall)).reshape(Ph, Pw)

    pred_full = np.zeros((H, W), dtype=np.uint8)
    for i in range(Ph):
        y0 = int(i * H / Ph); y1 = int((i + 1) * H / Ph)
        for j in range(Pw):
            x0 = int(j * W / Pw); x1 = int((j + 1) * W / Pw)
            pred_full[y0:y1, x0:x1] = pred_grid[i, j]

    valid = lbl_test > 0
    p = pred_full[valid]; t = lbl_test[valid]
    acc = float((p == t).mean())
    classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()))
    classes = [c for c in classes if c != 0]
    ious = {}
    for c in classes:
        inter = int(((p == c) & (t == c)).sum())
        union = int(((p == c) | (t == c)).sum())
        ious[int(c)] = inter / union if union else 0.0
    macro = float(np.mean(list(ious.values())))

    print(f"  test acc: {acc:.3f}")
    print(f"  per-class IoU:")
    for c in classes:
        print(f"    {ID_TO_DLTB.get(c, str(c)):<6}: {ious[c]:.3f}")
    print(f"  macro IoU: {macro:.3f}")

    # Save prediction as a GeoTIFF for visualisation.
    import rasterio
    with rasterio.open(test_data and Path(region_labels[len(TRAIN_BBOXES)][0])) as src:
        prof = src.profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0)
    for k in ("photometric", "interleave"): prof.pop(k, None)
    with rasterio.open(args.out_dir / "test_pred.tif", "w", **prof) as dst:
        dst.write(pred_full, 1)
    print(f"\n  saved prediction → {args.out_dir / 'test_pred.tif'}")


if __name__ == "__main__":
    main()
