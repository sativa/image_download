"""SAM3 INSTANCE-level parcel DELINEATION evaluation (the test CLAUDE.md mandates: 1m, parcel-level,
对标 FSDA / Delineate-Anything).

SAM3's text prompt ("farmland") returns PER-INSTANCE masks (shape (N,1,H,W)) — sam3_field_seg.py
threw the instances away with .any(0) (semantic union). Here we KEEP each instance = one predicted
field, build a predicted instance id-map, and score it against the DLTB cropland polygons (cid∈{1,2})
as DELINEATION (not classification): instance-match F1 (IoU≥0.5, panoptic/Delineate-Anything口径),
over-/under-segmentation, boundary-F1, and FSDA-style area agreement. This is SAM3's real strength
(instance segmentation), which was never actually benchmarked at parcel level before.

Run on .250 (sam3.pt + DLTB local, CUDA). c_1m x6 channel layout: [0:3]=Esri RGB, [3:6]=Google RGB.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
SAM3_REPO = "/home/ps/sam3/sam3-inference"


def load_processor(weights, device, conf, ft_state=""):
    sys.path.insert(0, SAM3_REPO)
    try:
        import decord  # noqa: F401
    except Exception:
        sys.modules["decord"] = __import__("types").ModuleType("decord")  # stub if missing
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=weights, load_from_HF=False, device=device)
    try:
        model.to(device).eval()                                    # build() ignores ":idx" devices -> force move
    except Exception as ex:
        print(f"  [warn] model.to({device}) failed: {ex}", flush=True)
    if ft_state:                                                   # load fine-tuned detector heads (sam3_finetune_b)
        st = torch.load(ft_state, map_location=device, weights_only=True)
        if "seg" in st:                                            # full {seg,trans,dps} (sam3_finetune_fast)
            model.segmentation_head.load_state_dict(st["seg"]); model.transformer.load_state_dict(st["trans"])
            if st.get("dps") is not None and getattr(model, "dot_prod_scoring", None) is not None:
                model.dot_prod_scoring.load_state_dict(st["dps"])
            print(f"  [ft] loaded full {{seg,trans,dps}} from {ft_state}", flush=True)
        else:                                                      # seg_head-only (older finetune_b: trans/dps stay zero-shot)
            model.segmentation_head.load_state_dict(st)
            print(f"  [ft] loaded seg_head-only from {ft_state} (trans/dps = original)", flush=True)
    return Sam3Processor(model, device=device, confidence_threshold=conf)


def _prompt_masks(proc, img_rgb, prompt):
    """One SAM3 text-prompt call -> (N, h, w) bool instance masks (empty -> (0,h,w))."""
    st = proc.set_image(Image.fromarray(img_rgb))
    st = proc.set_text_prompt(prompt=prompt, state=st)
    m = st.get("masks")
    if m is None or m.numel() == 0:
        return np.zeros((0, *img_rgb.shape[:2]), bool)
    return m.squeeze(1).cpu().numpy().astype(bool)


def _claim(idmap, masks, min_area_px, next_id, y0=0, x0=0, max_area_frac=0.4):
    """Claim masks onto still-free pixels (no overlap). SMALL-first so fine parcels keep their
    boundaries; DROP oversized masks (SAM3's "whole farmland" semantic blob, not a field). """
    h, w = masks.shape[1], masks.shape[2]
    cap = max_area_frac * h * w
    sub = idmap[y0:y0 + h, x0:x0 + w]
    areas = [int(masks[i].sum()) for i in range(masks.shape[0])]
    for k in sorted(range(masks.shape[0]), key=lambda i: areas[i]):    # small first
        if areas[k] > cap or areas[k] < min_area_px:                    # drop blob / drop noise
            continue
        claim = masks[k] & (sub == 0)
        if int(claim.sum()) < min_area_px:
            continue
        sub[claim] = next_id
        next_id += 1
    return next_id


def sam3_instances(proc, rgb, prompt, tile, min_area_px, resize_to=0):
    """rgb (H,W,3) uint8 -> instance id-map (H,W) int32 (0=unclaimed), n_inst.

    resize_to>0: down-scale the WHOLE cell to resize_to (SAM3's native res, full semantic context),
    build the id-map there, nearest-resize it back to (H,W) so it shares the DLTB grid. Else: grid-tile
    at native resolution (sharper boundaries, but tile splits = mild over-seg).
    """
    H, W = rgb.shape[:2]
    if resize_to:
        small = np.array(Image.fromarray(rgb).resize((resize_to, resize_to), Image.BILINEAR))
        masks = _prompt_masks(proc, small, prompt)
        idmap_s = np.zeros((resize_to, resize_to), np.int32)
        min_s = max(4, int(min_area_px * (resize_to / max(H, W)) ** 2))   # scale area threshold
        n = _claim(idmap_s, masks, min_s, 1) - 1
        idmap = cv2.resize(idmap_s, (W, H), interpolation=cv2.INTER_NEAREST)
        return idmap, n
    idmap = np.zeros((H, W), np.int32)
    next_id = 1
    wins = [(0, 0, H, W)] if not (tile and (H > tile or W > tile)) else [
        (y, x, min(y + tile, H), min(x + tile, W)) for y in range(0, H, tile) for x in range(0, W, tile)]
    for (y0, x0, y1, x1) in wins:
        masks = _prompt_masks(proc, rgb[y0:y1, x0:x1], prompt)
        next_id = _claim(idmap, masks, min_area_px, next_id, y0, x0)
    return idmap, next_id - 1


def load_classifier(cls_ckpt, device):
    """DINOv3-Sat 7-class head (dino_v3_bd, in_ch=11) — the OBIA classifier that filters SAM3 instances."""
    from train_dino_1m_v3 import DinoV3FreqUNetBD, DINOV3_SAT
    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBD(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(device)
    sd = torch.load(cls_ckpt, map_location=device, weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    return m


@torch.no_grad()
def cls_prob(model, x6, device, cs=448):
    """x6 (6,H,W) uint8 -> (9,H,W) class softmax (zero NDVI, as parcel_seg/parcel_bh do for eval)."""
    from train_dino_1m import norm6
    _, H, W = x6.shape
    ndvi = np.zeros((5, H, W), np.float32)
    acc = np.zeros((9, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - cs + 1), cs)); xs = list(range(0, max(1, W - cs + 1), cs))
    if ys[-1] != H - cs: ys.append(max(0, H - cs))
    if xs[-1] != W - cs: xs.append(max(0, W - cs))
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t + cs, l:l + cs]), ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb)[0]
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
    return acc / np.maximum(cnt, 1)


def filter_noncrop(pred, n_pred, cprob):
    """Drop SAM3 instances whose mean 7-class argmax is NOT cropland(1)/orchard(2) -> kills the
    over-prediction in non-cropland cells (the OBIA fix: SAM3 boundaries + classifier veto)."""
    flat = cprob[1:].reshape(8, -1)                              # classes 1..8 per pixel
    kept = 0
    for pid in range(1, n_pred + 1):
        m = pred == pid
        if not m.any():
            continue
        c = int(flat[:, m.ravel()].mean(1).argmax()) + 1         # 1..8 (8设施大棚->建筑, non-crop)
        if c not in (1, 2):
            pred[m] = 0                                          # veto non-cropland instance
        else:
            kept += 1
    return kept


def load_county(dltb, county, cache):
    if county in cache:
        return cache[county]
    g = gpd.read_parquet(Path(dltb) / f"{county}.parquet")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except AttributeError:
        g["geometry"] = g.geometry.buffer(0)
    cid = g["DLBM"].astype(str).str[:2]
    g["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0").astype(int)
    g = g[(g["cid"] >= 1) & (g["cid"] <= 2)].reset_index(drop=True)   # CROPLAND only (耕地+园地)
    cache[county] = g
    return g


def dltb_idmap(g, bbox, H, W):
    """Rasterize DLTB cropland polygons over the cell -> gt instance id-map (0=non-cropland)."""
    tr = from_bounds(*[float(b) for b in bbox], W, H)
    cb = shp_box(*[float(b) for b in bbox])
    idx = list(g.sindex.intersection(tuple(float(b) for b in bbox)))
    if not idx:
        return None, tr
    sub = g.iloc[idx].reset_index(drop=True)
    shapes = [(geom, j + 1) for j, geom in enumerate(sub.geometry) if geom.intersects(cb)]
    if not shapes:
        return None, tr
    return rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="int32"), tr


def iou_match(pred, gt, n_pred, n_gt, iou_thr=0.5):
    """Greedy 1-to-1 instance matching by IoU (panoptic/Delineate-Anything口径).

    Returns tp, and per-gt #pred-overlaps (over-seg) / per-pred #gt-overlaps (under-seg)."""
    if n_pred == 0 or n_gt == 0:
        return 0, [], []
    # intersection counts via joint histogram
    flat = pred.astype(np.int64) * (n_gt + 1) + gt.astype(np.int64)
    binc = np.bincount(flat.ravel(), minlength=(n_pred + 1) * (n_gt + 1))
    inter = binc.reshape(n_pred + 1, n_gt + 1)[1:, 1:]             # (P, G), drop bg row/col
    ap = np.bincount(pred.ravel(), minlength=n_pred + 1)[1:]
    ag = np.bincount(gt.ravel(), minlength=n_gt + 1)[1:]
    union = ap[:, None] + ag[None, :] - inter
    iou = inter / np.maximum(union, 1)
    # over-seg: per gt, how many pred instances overlap it meaningfully (>=20% of pred in this gt)
    overlap = inter > 0
    over = (overlap & (iou > 0.1)).sum(0)                          # #pred per gt
    under = (overlap & (iou > 0.1)).sum(1)                         # #gt per pred
    # greedy 1-1 by descending IoU
    tp = 0
    if iou.size:
        pairs = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        used_p = np.zeros(n_pred, bool); used_g = np.zeros(n_gt, bool)
        for pi, gi in pairs:
            if iou[pi, gi] < iou_thr:
                break
            if not used_p[pi] and not used_g[gi]:
                used_p[pi] = used_g[gi] = True; tp += 1
    return tp, over.tolist(), under.tolist()


def edges(idmap):
    """Boundary pixels of an instance id-map (where a 4-neighbour has a different id)."""
    e = np.zeros(idmap.shape, bool)
    e[:-1, :] |= idmap[:-1, :] != idmap[1:, :]
    e[1:, :] |= idmap[:-1, :] != idmap[1:, :]
    e[:, :-1] |= idmap[:, :-1] != idmap[:, 1:]
    e[:, 1:] |= idmap[:, :-1] != idmap[:, 1:]
    return e & (idmap > 0)


def bf1(pb, tb, tol=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    td = cv2.dilate(tb.astype(np.uint8), k) > 0; pd = cv2.dilate(pb.astype(np.uint8), k) > 0
    p = (pb & td).sum() / max(1, pb.sum()); r = (tb & pd).sum() / max(1, tb.sum())
    return 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    ap.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    ap.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    ap.add_argument("--prompt", default="farmland")
    ap.add_argument("--tile", type=int, default=1008)              # SAM3 native resolution (native-res tiling)
    ap.add_argument("--resize-to", type=int, default=1008, help="0=tile; >0=whole-cell resize to this (full context)")
    ap.add_argument("--conf", type=float, default=0.25)            # 0.25 farmland: ~70 fields/cell, 95% cov, de-noised
    ap.add_argument("--min-area-px", type=int, default=64)
    ap.add_argument("--n-cells", type=int, default=120)
    ap.add_argument("--only", default="", help="comma-sep cell names to eval (diagnostic)")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--viz", default="", help="dir to dump RGB|GT-instances|SAM3-instances composites")
    ap.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_parcel_eval.json")
    ap.add_argument("--ft-state", default="", help="sam3_finetune_b ft_state.pt -> load fine-tuned detector heads")
    ap.add_argument("--cls-ckpt", default="", help="DINOv3-Sat 7-class head (dino_v3_bd) -> OBIA filter non-cropland instances")
    a = ap.parse_args()
    t0 = time.time()
    proc = load_processor(a.weights, a.device, a.conf, a.ft_state)
    classifier = load_classifier(a.cls_ckpt, a.device) if a.cls_ckpt else None
    if classifier is not None:
        print(f"  [obia] loaded classifier {a.cls_ckpt} -> will veto non-cropland SAM3 instances", flush=True)
    print(f"[sam3-parcel] model loaded ({time.time()-t0:.0f}s) | prompt='{a.prompt}' tile={a.tile} conf={a.conf}", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()]
    if a.only:
        want = set(a.only.split(","))
        te = [n for n in te if n in want]
    else:
        te = te[:a.n_cells]
    cache = {}
    if a.viz:
        Path(a.viz).mkdir(parents=True, exist_ok=True)
    TP = FP = FN = 0
    over_all = []; under_all = []
    bf1s = []
    pred_area_m = true_area_m = 0.0
    parcels = []   # (gt_area_m2, matched_bool) for MMU sweep on recall
    nc = 0
    for i, n in enumerate(te):
        z = np.load(Path(a.data_dir) / f"{n}.npz"); x6 = z["x6"]; bbox = z["bbox"]
        _, H, W = x6.shape
        pix_m = (float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(
            math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W
        rgb = np.ascontiguousarray(x6[:3].transpose(1, 2, 0)).astype(np.uint8)
        try:
            g = load_county(a.dltb, n.split("_")[0], cache)
        except Exception as ex:
            print(f"  skip {n}: {ex}", flush=True); continue
        gt, tr = dltb_idmap(g, bbox, H, W)
        if gt is None:
            continue
        n_gt = int(gt.max())
        pred, n_pred = sam3_instances(proc, rgb, a.prompt, a.tile, a.min_area_px, a.resize_to)
        if classifier is not None:                                  # OBIA: veto non-cropland instances
            filter_noncrop(pred, n_pred, cls_prob(classifier, x6, a.device))
            ids = np.unique(pred); ids = ids[ids > 0]               # relabel survivors 1..K (no FP from holes)
            remap = np.zeros(int(pred.max()) + 1, np.int32) if pred.max() else np.zeros(1, np.int32)
            if ids.size:
                remap[ids] = np.arange(1, ids.size + 1)
            pred = remap[pred]; n_pred = int(ids.size)
        tp, over, under = iou_match(pred, gt, n_pred, n_gt)
        if a.only:
            ga = np.bincount(gt.ravel())[1:] if n_gt else np.array([])
            pa = np.bincount(pred.ravel())[1:] if n_pred else np.array([])
            print(f"  [{n}] n_pred={n_pred} n_gt={n_gt} TP={tp} | gt_area_px med={int(np.median(ga)) if ga.size else 0} "
                  f"max={int(ga.max()) if ga.size else 0} | pred_area_px med={int(np.median(pa)) if pa.size else 0} "
                  f"max={int(pa.max()) if pa.size else 0} | pred_cov={(pred>0).mean()*100:.0f}% gt_cov={(gt>0).mean()*100:.0f}%", flush=True)
        TP += tp; FP += (n_pred - tp); FN += (n_gt - tp)
        over_all += over; under_all += under
        bf1s.append(bf1(edges(pred), edges(gt), 3))
        pred_area_m += float((pred > 0).sum()) * pix_m * pix_m
        true_area_m += float((gt > 0).sum()) * pix_m * pix_m
        # per-gt matched? (for MMU recall sweep): a gt is matched if some pred has IoU>=0.5 — approx by over>0 & ...
        # recompute matched gts precisely is in iou_match; here use area sweep on recall via gt sizes
        for j in range(1, n_gt + 1):
            ga = int((gt == j).sum())
            if ga:
                parcels.append((ga * pix_m * pix_m, over[j - 1] if j - 1 < len(over) else 0))
        nc += 1
        if a.viz and i < 12:
            _viz(rgb, gt, pred, Path(a.viz) / f"{n}.png")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(te)} cells | pred_inst≈{n_pred} gt_inst={n_gt} ({time.time()-t0:.0f}s)", flush=True)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9), pr, rc
    F, P, R = f1(TP, FP, FN)
    overm = float(np.mean([o for o in over_all if o > 0])) if over_all else 0
    underm = float(np.mean([u for u in under_all if u > 0])) if under_all else 0
    agree = 100.0 * min(pred_area_m, true_area_m) / (max(pred_area_m, true_area_m) + 1e-9)
    print(f"\n[sam3-parcel] === {nc} cells | SAM3 instance delineation vs DLTB cropland ===", flush=True)
    print(f"  INSTANCE-match F1 (IoU>=0.5) = {F:.4f}  (P{P:.3f}/R{R:.3f})   TP{TP} FP{FP} FN{FN}", flush=True)
    print(f"  boundary-F1 (tol3px)         = {float(np.mean(bf1s)):.4f}", flush=True)
    print(f"  over-seg (preds per matched gt)  = {overm:.2f}   under-seg (gts per pred) = {underm:.2f}", flush=True)
    print(f"  AREA-MATCH (FSDA口径) = {agree:.1f}%  (pred/ref={pred_area_m/(true_area_m+1e-9):.3f}; FSDA Tibet 89.8%)", flush=True)
    print(f"  MMU(ha) recall sweep (gt matched if >=1 overlapping pred):", flush=True)
    out = {"weights": a.weights, "prompt": a.prompt, "instance_f1": F, "P": P, "R": R,
           "boundary_f1": float(np.mean(bf1s)), "over_seg": overm, "under_seg": underm,
           "area_agreement": agree, "n_cells": nc, "sweep": {}}
    for ha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        thr = ha * 10000
        sel = [(ar, ov) for ar, ov in parcels if ar >= thr]
        if not sel:
            continue
        rec = sum(1 for _, ov in sel if ov > 0) / len(sel)
        print(f"    >={ha:<5} n={len(sel):>6d}  detected={rec:.3f}", flush=True)
        out["sweep"][ha] = {"n": len(sel), "gt_detected": rec}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"  -> {a.out}", flush=True)


def _viz(rgb, gt, pred, path, max_side=900):
    """RGB | GT instances (random colours) | SAM3 instances."""
    def colorize(idm):
        rng = (idm.astype(np.uint32) * 2654435761) % (1 << 24)
        c = np.zeros((*idm.shape, 3), np.uint8)
        c[..., 0] = (rng & 255); c[..., 1] = ((rng >> 8) & 255); c[..., 2] = ((rng >> 16) & 255)
        c[idm == 0] = 0
        return c
    H, W = rgb.shape[:2]; s = max(1, max(H, W) // max_side)
    r = rgb[::s, ::s]; gc = colorize(gt[::s, ::s]); pc = colorize(pred[::s, ::s])
    gov = (0.5 * r + 0.5 * gc).astype(np.uint8); pov = (0.5 * r + 0.5 * pc).astype(np.uint8)
    gap = np.full((r.shape[0], 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([r, gap, gov, gap, pov], 1)).save(path)


if __name__ == "__main__":
    main()
