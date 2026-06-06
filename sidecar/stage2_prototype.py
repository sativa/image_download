"""Stage 2 prototype: v25 (S2 EfNet) + SAM + Grounding DINO ensemble.

Pipeline per test cell:
  1. v25 inference on S2 → binary cropland mask at S2 10m grid (240x240)
  2. Upsample v25 mask to z17 1m grid (~2304x2560)
  3. Three Stage-2 variants on z17 RGB:
     (a) SAM-points: sample N points inside v25 mask, SAM predicts mask
     (b) GroundingDINO: text prompt "agricultural field" → boxes
         filter boxes overlapping v25 mask, union all boxes → mask
     (c) GD+SAM: each GD box → SAM mask, union all → final mask
  4. Ensemble: majority vote across (a), (b), (c), plus baseline v25 upsampled
  5. Compare each variant to 三调 GT at z17 grid → F1

Evaluation: 8 test cells from v17_regions_balanced.test, both esri and google.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS
from train_v16_binary import rasterise_dltb_binary

S2_DIR = HOME / "data/v19_s2_raw"
Z17_DIR = HOME / "data/v11_imagery"
DLTB_DIR = HOME / "data/v11_dltb"

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5
NDVI_STD = 0.3


def load_v25(ckpt, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                     in_channels=5, classes=3).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return getattr(model, "eval")()


def v25_predict_s2(model, rgbnir, ndvi, device):
    """Run v25 on S2 → returns (H, W) binary cropland prob ∈ [0,1]."""
    x = rgbnir.astype(np.float32).copy()
    for b in range(4):
        x[b] = (x[b] - S2_MEAN[b]) / S2_STD[b]
    ndvi_n = (ndvi.astype(np.float32) - NDVI_MEAN) / NDVI_STD
    x5 = np.concatenate([x, ndvi_n[None, ...]], axis=0)[None, ...]
    xt = torch.from_numpy(x5.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(xt)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    return prob


def upsample_to(arr, H, W):
    t = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
    out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def sam_with_points(sam, rgb_z17, prob_mask, n_points=24):
    """Run SAM on z17 RGB with point prompts sampled from prob_mask."""
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_z17)
    # Sample point prompts: top-N positives from high-confidence regions
    H, W = prob_mask.shape
    flat = prob_mask.ravel()
    # Spread points: pick random within prob > 0.6
    pos_idx = np.flatnonzero(flat > 0.6)
    if len(pos_idx) < n_points:
        # Not enough high-confidence — empty result
        return np.zeros((H, W), dtype=bool)
    sampled = np.random.choice(pos_idx, size=n_points, replace=False)
    ys = sampled // W; xs = sampled % W
    point_coords = np.stack([xs, ys], axis=1).astype(np.float32)
    point_labels = np.ones(n_points, dtype=np.int32)  # all foreground
    masks, _, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, multimask_output=False)
    return masks[0]  # H x W bool


def grounding_dino_boxes(processor, gd_model, rgb_z17, prompt, device):
    """Run Grounding DINO → list of boxes (x1,y1,x2,y2) at z17 grid."""
    from PIL import Image
    pil = Image.fromarray(rgb_z17)
    inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    H, W = rgb_z17.shape[:2]
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
        target_sizes=[(H, W)])[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    return boxes, scores


def sam_with_box(sam, rgb_z17_set, box):
    """SAM with a single box prompt."""
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_z17_set)
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    return masks[0]


def metrics_binary(pred, truth):
    valid = truth > 0
    if not valid.any(): return None
    p = pred[valid]; t = truth[valid]
    pi = p == 1; ti = t == 1
    tp = int((pi & ti).sum())
    fp = int((pi & ~ti).sum())
    fn = int((~pi & ti).sum())
    tn = int((~pi & ~ti).sum())
    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    iou = tp/(tp+fp+fn) if tp+fp+fn else 0
    acc = (tp+tn)/max(tp+fp+fn+tn, 1)
    return {"acc": acc, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--v25-ckpt", type=Path, default=HOME / "results/v25/best.pt")
    p.add_argument("--sam-ckpt", type=Path, default=Path.home() / ".cache/sam/sam_vit_h_4b8939.pth")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-points", type=int, default=24)
    p.add_argument("--out-dir", type=Path, default=HOME / "results/stage2")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load v25
    print("[1] load v25 ...", flush=True)
    v25 = load_v25(args.v25_ckpt, device)

    # Load SAM
    print("[2] load SAM-H ...", flush=True)
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_ckpt))
    sam = sam.to(device).eval()

    # Load Grounding DINO
    print("[3] load Grounding DINO ...", flush=True)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gd_id = "IDEA-Research/grounding-dino-tiny"
    gd_processor = AutoProcessor.from_pretrained(gd_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(device).eval()

    # Load test cells
    print("[4] load test cells ...", flush=True)
    regions = json.loads(args.regions_json.read_text())
    test = regions["test"]
    import geopandas as gpd
    import rasterio
    gdf = {}
    for r in test:
        c = r["county"]
        if c in gdf: continue
        g = gpd.read_parquet(DLTB_DIR / f"{c}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf[c] = g

    test_cells = []
    for r in test:
        bb = tuple(r["bbox"])
        # Pick esri only (one z17 source per cell to start)
        path = Z17_DIR / f"{r['county']}_{r['idx']}_esri.tif"
        if not path.exists(): continue
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            transform = rs.transform
            H, W = rs.height, rs.width
        label = rasterise_dltb_binary(gdf[r["county"]], bb, transform, H, W)
        # Convert label {0:nodata, 1:crop, 2:other} → {1:crop, 0:other-or-nodata}
        s2_path = S2_DIR / f"{r['county']}_{r['idx']}.npz"
        if not s2_path.exists(): continue
        s2 = np.load(s2_path)
        test_cells.append({
            "name": f"{r['county']}_{r['idx']}",
            "rgb": rgb, "H": H, "W": W,
            "rgbnir": s2["rgbnir"], "ndvi": s2["ndvi"],
            "label": label,  # 0 nodata, 1 crop, 2 other
        })
    print(f"  loaded {len(test_cells)} test cells", flush=True)

    # Run all variants
    print(f"\n[5] running 4 variants on {len(test_cells)} cells ...", flush=True)
    results = {v: [] for v in ["v25_baseline", "v25_sam", "v25_gd", "v25_gd_sam", "ensemble"]}
    NONE_METRIC = {"acc": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for cell in test_cells:
        t0 = time.time()
        name = cell["name"]; H, W = cell["H"], cell["W"]
        # Stage 1: v25 inference on S2
        prob_s2 = v25_predict_s2(v25, cell["rgbnir"], cell["ndvi"], device)
        prob_z17 = upsample_to(prob_s2, H, W)
        v25_mask = (prob_z17 > 0.5).astype(np.uint8)

        # (a) v25 baseline
        m_base = metrics_binary(v25_mask, cell["label"]) or dict(NONE_METRIC)
        results["v25_baseline"].append(m_base)

        # (b) SAM with points sampled from v25 mask
        sam_mask = v25_mask.astype(bool)
        m_sam = dict(NONE_METRIC)
        try:
            sam_mask = sam_with_points(sam, cell["rgb"], prob_z17, n_points=args.n_points)
            m_sam = metrics_binary(sam_mask.astype(np.uint8), cell["label"]) or dict(NONE_METRIC)
        except Exception as e:
            print(f"  SAM failed on {name}: {str(e)[:80]}", flush=True)
        results["v25_sam"].append(m_sam)

        # (c) Grounding DINO boxes → mask
        boxes = []
        gd_mask = np.zeros((H, W), dtype=np.uint8)
        m_gd = dict(NONE_METRIC)
        try:
            prompt = "agricultural field. farmland. crop field. cultivated land."
            boxes, scores = grounding_dino_boxes(gd_processor, gd_model, cell["rgb"],
                                                   prompt, device)
            for b in boxes:
                x1, y1, x2, y2 = b.astype(int)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                gd_mask[y1:y2, x1:x2] = 1
            m_gd = metrics_binary(gd_mask, cell["label"]) or dict(NONE_METRIC)
        except Exception as e:
            print(f"  GD failed on {name}: {str(e)[:80]}", flush=True)
        results["v25_gd"].append(m_gd)

        # (d) GD + SAM (SAM per box)
        gd_sam_mask = np.zeros((H, W), dtype=bool)
        m_gd_sam = dict(NONE_METRIC)
        try:
            from segment_anything import SamPredictor
            if len(boxes) > 0:
                predictor = SamPredictor(sam)
                predictor.set_image(cell["rgb"])
                for b in boxes:
                    masks, _, _ = predictor.predict(box=b, multimask_output=False)
                    gd_sam_mask |= masks[0]
            m_gd_sam = metrics_binary(gd_sam_mask.astype(np.uint8), cell["label"]) or dict(NONE_METRIC)
        except Exception as e:
            print(f"  GD+SAM failed on {name}: {str(e)[:80]}", flush=True)
        results["v25_gd_sam"].append(m_gd_sam)

        # (e) Ensemble: 2-out-of-3 vote across {v25, SAM, GD+SAM}
        votes = v25_mask.astype(int) + sam_mask.astype(int) + gd_sam_mask.astype(int)
        ens_mask = (votes >= 2).astype(np.uint8)
        m_ens = metrics_binary(ens_mask, cell["label"]) or dict(NONE_METRIC)
        results["ensemble"].append(m_ens)
        print(f"  {name}: v25={m_base['f1']:.3f} sam={m_sam['f1']:.3f} "
              f"gd={m_gd['f1']:.3f} gd_sam={m_gd_sam['f1']:.3f} ens={m_ens['f1']:.3f} "
              f"n_boxes={len(boxes)} ({time.time()-t0:.0f}s)", flush=True)

    # Aggregate
    print(f"\n[6] aggregate", flush=True)
    print(f"{'Variant':<20} {'F1':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}")
    out_summary = {}
    for v, ms in results.items():
        if not ms: continue
        avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"{v:<20} {avg['f1']:<8.3f} {avg['iou']:<8.3f} {avg['precision']:<8.3f} {avg['recall']:<8.3f}")
        out_summary[v] = avg
    (args.out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"\nwrote {args.out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
