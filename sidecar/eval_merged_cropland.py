"""Re-evaluate v14 with 园地 merged into 耕地 (broad cropland).

The merged 4-class scheme:
  1 = 广义耕地 (formerly 耕地 + 园地)
  3 = 林地
  4 = 草地
  5 = 其他

Reports per-class IoU + accuracy + recall + precision for 广义耕地,
under both single-pass and TTA. Uses v14's best.pt.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import (
    DEFAULT_DINOV2, DLTB_CLASS_TO_ID, ID_TO_DLTB, IMAGENET_MEAN, IMAGENET_STD,
    DLBM_TO_CLASS, rasterise_dltb_region, DinoUNet,
)
from eval_tta import slide_predict_softmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=HOME / "results/v14/best.pt")
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v11_regions_balanced.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--use-tta", action="store_true", help="enable 4-flip TTA")
    args = p.parse_args()
    device = args.device
    flips = ["identity", "h", "v", "hv"] if args.use_tta else ["identity"]

    print(f"[1] Loading model + test data", flush=True)
    import geopandas as gpd
    import rasterio
    from transformers import AutoModel

    regions = json.loads(args.regions_json.read_text())
    gdf = {}
    for r in regions["test"]:
        if r["county"] in gdf:
            continue
        g = gpd.read_parquet(args.dltb_cache / f"{r['county']}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf[r["county"]] = g

    test = []
    for r in regions["test"]:
        bb = tuple(r["bbox"])
        for src in ["esri", "google"]:
            path = args.data_cache / f"{r['county']}_{r['idx']}_{src}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as rs:
                bands = rs.read(out_dtype="uint8")
                transform = rs.transform; H, W = rs.height, rs.width
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            lbl = rasterise_dltb_region(gdf[r["county"]], bb, transform, H, W)
            test.append((f"{r['county']}_{r['idx']}_{src}", rgb, lbl))
    print(f"  {len(test)} test images, TTA={args.use_tta}", flush=True)

    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=6, unfreeze_last_n=4).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = getattr(model, "eval")()

    print(f"\n[2] Inference + merged-class evaluation", flush=True)
    # Aggregate confusion across all cells.
    # Original 5 classes; we'll re-bin to 4 after.
    all_p = []
    all_t = []
    t0 = time.time()
    for name, rgb, lbl in test:
        if args.use_tta:
            avg_probs = np.zeros((rgb.shape[0], rgb.shape[1], 6), dtype=np.float32)
            for flip in flips:
                f = None if flip == "identity" else flip
                avg_probs += slide_predict_softmax(model, rgb, device, flip=f)
            avg_probs /= len(flips)
            pred = avg_probs.argmax(axis=-1).astype(np.uint8)
        else:
            pred = slide_predict_softmax(model, rgb, device).argmax(axis=-1).astype(np.uint8)
        valid = lbl > 0
        if valid.any():
            all_p.append(pred[valid])
            all_t.append(lbl[valid])
    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t)
    print(f"  inference done in {time.time()-t0:.0f}s, {len(all_p):,} labelled pixels", flush=True)

    # === Original 5-class metrics ===
    print(f"\n[3] Original 5-class IoU (for reference)", flush=True)
    for c in range(1, 6):
        inter = int(((all_p == c) & (all_t == c)).sum())
        union = int(((all_p == c) | (all_t == c)).sum())
        iou = inter / union if union else 0
        print(f"  {ID_TO_DLTB[c]:<4} (id={c}): IoU={iou:.3f}")
    overall_acc = (all_p == all_t).mean()
    print(f"  overall pixel acc: {overall_acc:.3f}")

    # === Merged: 园地 → 耕地 (id 2 → 1) ===
    print(f"\n[4] Merged 4-class metrics (园地 collapsed into 耕地)", flush=True)
    pred_m = np.where(all_p == 2, 1, all_p)
    truth_m = np.where(all_t == 2, 1, all_t)
    # Per-class
    classes_m = sorted(set(np.unique(pred_m).tolist()) | set(np.unique(truth_m).tolist()))
    classes_m = [c for c in classes_m if c != 0]
    LABELS_M = {1: "广义耕地", 3: "林地", 4: "草地", 5: "其他"}
    ious_m = {}
    for c in classes_m:
        inter = int(((pred_m == c) & (truth_m == c)).sum())
        union = int(((pred_m == c) | (truth_m == c)).sum())
        ious_m[c] = inter / union if union else 0
        print(f"  {LABELS_M.get(c, c):<6} (id={c}): IoU={ious_m[c]:.3f}")
    macro_m = float(np.mean(list(ious_m.values())))
    overall_acc_m = (pred_m == truth_m).mean()
    print(f"  macro IoU (4 classes): {macro_m:.3f}")
    print(f"  overall pixel acc:    {overall_acc_m:.3f}")

    # === Focused: 广义耕地 recall / precision / F1 ===
    print(f"\n[5] 广义耕地 detection (binary: cropland-broad vs other)", flush=True)
    is_pred_crop = (pred_m == 1)
    is_truth_crop = (truth_m == 1)
    tp = int((is_pred_crop & is_truth_crop).sum())
    fp = int((is_pred_crop & ~is_truth_crop).sum())
    fn = int((~is_pred_crop & is_truth_crop).sum())
    tn = int((~is_pred_crop & ~is_truth_crop).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    print(f"  pixels labelled as 广义耕地 in ground truth: {tp+fn:,} ({(tp+fn)/len(all_p)*100:.1f}%)")
    print(f"  pixels predicted as 广义耕地:                {tp+fp:,}")
    print(f"  recall    (TP / (TP+FN)): {recall:.3f}  ← 'of all cropland pixels, how many caught'")
    print(f"  precision (TP / (TP+FP)): {precision:.3f}  ← 'of all predicted cropland, how many real'")
    print(f"  F1 score:                  {f1:.3f}")
    print(f"  IoU:                       {iou:.3f}")
    print()
    print(f"  ★ '耕地识别精度 ≥ 80%' interpretation:")
    print(f"     - recall ≥ 0.80:  {'✅ YES' if recall >= 0.8 else '❌ NO'} (current: {recall:.3f})")
    print(f"     - precision ≥ 0.80: {'✅ YES' if precision >= 0.8 else '❌ NO'} (current: {precision:.3f})")
    print(f"     - F1 ≥ 0.80:      {'✅ YES' if f1 >= 0.8 else '❌ NO'} (current: {f1:.3f})")


if __name__ == "__main__":
    main()
