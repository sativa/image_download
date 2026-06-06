"""SAM3 zero-shot field/parcel segmentation on 1m imagery -> cropland mask.

Tests the user's 1m-PRIMARY vision: prompt SAM3 (open-vocabulary, text-promptable) with
"farmland"/"field" on the 1m Esri tiles (c_1m), union the instance masks into a binary
cropland prediction, score pixel F1/IoU vs the 1m DLTB label (1=cropland). NO fine-tuning.

This is the premise check: does a foundation segmenter zero-shot transfer to overhead 1m
imagery for agricultural parcels? If yes -> fine-tune + OBIA-classify per parcel next.

Run on .250 (sam3.pt is local). Channel layout of c_1m x6: [0:3]=Esri RGB, [3:6]=Google RGB.
"""
import argparse, json, sys, time, types
from pathlib import Path

import numpy as np
from PIL import Image
import torch

SAM3_REPO = "/home/ps/sam3/sam3-inference"


def load_processor(weights, device, conf):
    sys.path.insert(0, SAM3_REPO)
    try:
        import decord  # noqa: F401  (real install present; stub is a fallback)
    except Exception:
        m = types.ModuleType("decord")
        m.cpu = m.gpu = lambda *a, **k: None
        m.VideoReader = object
        m.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
        sys.modules["decord"] = m
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=weights, load_from_HF=False, device=device)
    return Sam3Processor(model, device=device, confidence_threshold=conf)


def field_mask(proc, rgb, prompt, tile):
    """rgb (H,W,3) uint8 -> (H,W) bool union of `prompt` instance masks. n_inst total."""
    H, W = rgb.shape[:2]
    pred = np.zeros((H, W), bool)
    n_inst = 0
    wins = [(0, 0, H, W)]
    if tile and (H > tile or W > tile):
        wins = [(y, x, min(y + tile, H), min(x + tile, W))
                for y in range(0, H, tile) for x in range(0, W, tile)]
    for (y0, x0, y1, x1) in wins:
        st = proc.set_image(Image.fromarray(rgb[y0:y1, x0:x1]))
        st = proc.set_text_prompt(prompt=prompt, state=st)
        m = st.get("masks")
        if m is not None and m.numel() > 0:
            n_inst += m.shape[0]
            pred[y0:y1, x0:x1] |= m.squeeze(1).any(0).cpu().numpy()
    return pred, n_inst


def save_viz(rgb, lbl, pred, path, max_side=900):
    """Composite: RGB | GT cropland (green) | SAM3 pred (red). Downscaled for size."""
    H, W = rgb.shape[:2]
    s = max(1, max(H, W) // max_side)
    r = rgb[::s, ::s]; lb = lbl[::s, ::s]; pr = pred[::s, ::s]
    gt_ov = r.copy(); gt_ov[lb == 1] = (0.5 * gt_ov[lb == 1] + np.array([0, 128, 0])).astype(np.uint8)
    pr_ov = r.copy(); pr_ov[pr] = (0.5 * pr_ov[pr] + np.array([180, 0, 0])).astype(np.uint8)
    gap = np.full((r.shape[0], 8, 3), 255, np.uint8)
    comp = np.concatenate([r, gap, gt_ov, gap, pr_ov], axis=1)
    Image.fromarray(comp).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--regions-json", default="/mnt/sda/zf/landform/data/c_1m/manifest.json")
    p.add_argument("--split", default="test")
    p.add_argument("--n-cells", type=int, default=3)
    p.add_argument("--prompts", nargs="+",
                   default=["farmland", "field", "agricultural field", "crop field"])
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--tile", type=int, default=0, help="0=whole cell; else sub-tile px")
    p.add_argument("--src-ch", type=int, default=0, help="0=Esri RGB, 3=Google RGB")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_field")
    p.add_argument("--device", default="cuda")
    p.add_argument("--viz-n", type=int, default=2)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    R = json.loads(Path(a.regions_json).read_text())
    dd = Path(a.data_dir)
    to_name = lambda e: e if isinstance(e, str) else f"{e['county']}_{e['idx']}"
    names_all = [n for n in (to_name(e) for e in R[a.split]) if (dd / f"{n}.npz").exists()]
    step = max(1, len(names_all) // a.n_cells)  # spread across counties, not just the first
    names = names_all[::step][:a.n_cells]
    print(f"[sam3-field] {len(names)} {a.split} cells | prompts={a.prompts} | tile={a.tile} | conf={a.conf}", flush=True)

    t0 = time.time()
    proc = load_processor(a.weights, a.device, a.conf)
    print(f"[sam3-field] model loaded ({time.time()-t0:.0f}s)", flush=True)

    agg = {pr: dict(tp=0, fp=0, fn=0, inst=0) for pr in a.prompts}
    for ci, name in enumerate(names):
        z = np.load(dd / f"{name}.npz")
        rgb = np.ascontiguousarray(z["x6"][a.src_ch:a.src_ch + 3].transpose(1, 2, 0))
        lbl = z["label"]
        gt = lbl == 1; valid = lbl > 0
        tc = time.time()
        line = [f"  [{ci+1}/{len(names)}] {name} ({valid.mean()*100:.0f}% valid, crop {gt.sum()/max(valid.sum(),1)*100:.0f}%)"]
        for pr in a.prompts:
            pred, ninst = field_mask(proc, rgb, pr, a.tile)
            tp = int((pred & gt).sum()); fp = int((pred & valid & ~gt).sum()); fn = int((~pred & gt).sum())
            agg[pr]["tp"] += tp; agg[pr]["fp"] += fp; agg[pr]["fn"] += fn; agg[pr]["inst"] += ninst
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            line.append(f"    '{pr}': F1={f1:.3f} inst={ninst}")
            if ci < a.viz_n:
                save_viz(rgb, lbl, pred, out / f"viz_{name}_{pr.replace(' ','_')}.png")
        print("\n".join(line) + f"    ({time.time()-tc:.0f}s)", flush=True)

    print(f"\n[sam3-field] === aggregate over {len(names)} cells ===", flush=True)
    res = {}
    for pr in a.prompts:
        d = agg[pr]; tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9); iou = tp / max(tp + fp + fn, 1)
        res[pr] = dict(f1=f1, iou=iou, prec=prec, rec=rec, inst=d["inst"])
        print(f"  '{pr}': F1={f1:.4f} IoU={iou:.4f} P={prec:.3f} R={rec:.3f} inst={d['inst']}", flush=True)
    json.dump({"cells": names, "tile": a.tile, "conf": a.conf, "results": res},
              open(out / "sam3_field_result.json", "w"), indent=2)
    print(f"[sam3-field] done ({time.time()-t0:.0f}s) -> {out}/sam3_field_result.json", flush=True)


if __name__ == "__main__":
    main()
