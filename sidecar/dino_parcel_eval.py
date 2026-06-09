"""DINOv3-Sat distance-head DELINEATION (dist-peak watershed) — the PURE-DINO delineation line
(no SAM3), to compare head-to-head against SAM3-OBIA at parcel level.

dino_v3_bdd model has 3 heads: (cls 9-class, boundary, distance). Recipe (ResUNet-a/BsiNet):
distance-map local maxima = parcel centres -> watershed seeds -> instances; the cls head labels each
instance; keep cropland(1)/orchard(2). Object-level eval vs DLTB cropland polygons reuses the exact
metrics from sam3_parcel_eval (instance-match F1 / boundary-F1 / over-under-seg / area-match / MMU).
Run on .250 (CUDA). c_1m x6 (6,H,W) uint8 [0:3]=Esri [3:6]=Google; zero NDVI (as parcel_bh eval)."""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from sam3_parcel_eval import load_county, dltb_idmap, iou_match, edges, bf1, _viz
from train_dino_1m import norm6
from train_dino_1m_v3 import DinoV3FreqUNetBDD, DINOV3_SAT


@torch.no_grad()
def infer_heads(model, x6, dev, cs=448):
    """tiled -> (cls 9-class softmax (9,H,W), distance sigmoid (H,W)). zero NDVI."""
    _, H, W = x6.shape
    ndvi = np.zeros((5, H, W), np.float32)
    acc = np.zeros((9, H, W), np.float32); accd = np.zeros((H, W), np.float32)
    accb = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    st = max(1, cs // 2)                                           # 50% overlap
    ys = list(range(0, max(1, H - cs + 1), st)); xs = list(range(0, max(1, W - cs + 1), st))
    if ys[-1] != H - cs: ys.append(max(0, H - cs))
    if xs[-1] != W - cs: xs.append(max(0, W - cs))
    hw = np.hanning(cs)                                            # Hann window: 1 at centre -> 0 at edges
    win = np.maximum(np.outer(hw, hw), 1e-3).astype(np.float32)    # 2D weight; floor avoids /0 at image corners
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t + cs, l:l + cs]), ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            import contextlib
            _amp = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if str(dev).startswith("cuda") else contextlib.nullcontext())   # MPS/CPU: no cuda autocast
            with _amp:
                _o = model(xb); cls_lg, bnd_lg, dist_lg = _o[0], _o[1], _o[2]   # BDDF returns 4 (incl frame field)
                if cls_lg.shape[-2:] != (cs, cs):
                    cls_lg = F.interpolate(cls_lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    bnd_lg = F.interpolate(bnd_lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    dist_lg = F.interpolate(dist_lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(cls_lg.float(), 1)[0].cpu().numpy()
                pd = torch.sigmoid(dist_lg.float())[0, 0].cpu().numpy()
                pb = torch.sigmoid(bnd_lg.float())[0, 0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr * win; accd[t:t + cs, l:l + cs] += pd * win   # Hann-weighted blend
            accb[t:t + cs, l:l + cs] += pb * win; cnt[t:t + cs, l:l + cs] += win
    cnt = np.maximum(cnt, 1e-6)
    return acc / cnt, accd / cnt, accb / cnt


def dist_peak_instances(clsprob, dist, bnd, min_dist, peak_thr, min_area, use_ridge=False, downscale=1):
    """dist-peak watershed within the cropland(argmax∈{1,2}) mask -> instance id-map (relabelled 1..K).
    use_ridge: flood over a ridge = max(boundary-prob, 1-dist) so the boundary head reinforces edges.
    downscale>1: run the (expensive) watershed on a /downscale grid then NEAREST-upsample the labels —
    lets a single large mosaic be segmented in one pass (continuous, NO cell seams) in minutes, not >30min."""
    crop = np.isin(clsprob[1:].argmax(0) + 1, [1, 2])              # cropland+orchard pixels
    H, W = dist.shape
    if downscale > 1:
        hs, ws = max(1, H // downscale), max(1, W // downscale)
        dist_w = cv2.resize(dist, (ws, hs), interpolation=cv2.INTER_AREA)
        crop_w = cv2.resize(crop.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0
        bnd_w = cv2.resize(bnd, (ws, hs), interpolation=cv2.INTER_AREA)
        md, ma = max(2, min_dist // downscale), max(4, min_area // (downscale * downscale))
    else:
        dist_w, crop_w, bnd_w, md, ma = dist, crop, bnd, min_dist, min_area
    coords = peak_local_max(dist_w * crop_w, min_distance=md, threshold_abs=peak_thr)
    markers = np.zeros(dist_w.shape, np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    if markers.max() == 0:
        return np.zeros((H, W), np.int32), 0
    elevation = np.maximum(bnd_w, 1.0 - dist_w) if use_ridge else (-dist_w)
    inst = watershed(elevation, markers, mask=crop_w)              # flood from peaks to (boundary-reinforced) edges
    cnt = np.bincount(inst.ravel())
    small = np.nonzero(cnt < ma)[0]; small = small[small != 0]
    if small.size:
        inst[np.isin(inst, small)] = 0
    if downscale > 1:
        inst = cv2.resize(inst.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    ids = np.unique(inst); ids = ids[ids > 0]
    remap = np.zeros(int(inst.max()) + 1, np.int32) if inst.max() else np.zeros(1, np.int32)
    if ids.size:
        remap[ids] = np.arange(1, ids.size + 1)
    return remap[inst], int(ids.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_v3_bdd/best.pt")
    ap.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_lc")
    ap.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    ap.add_argument("--n-cells", type=int, default=120)
    ap.add_argument("--prefix", default="", help="glob cells by prefix (e.g. 620123_ for whole county eval)")
    ap.add_argument("--min-dist", type=int, default=10, help="peak_local_max min spacing (px)")
    ap.add_argument("--peak-thr", type=float, default=0.3, help="min distance-value for a seed")
    ap.add_argument("--min-area-px", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ridge", action="store_true", help="watershed over max(boundary,1-dist) ridge (vs -dist)")
    ap.add_argument("--dltb-parquet", default="", help="single DLTB parquet for all cells (cross-domain, e.g. Changzhi)")
    ap.add_argument("--viz", default="")
    ap.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_parcel_eval.json")
    a = ap.parse_args()
    t0 = time.time()
    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBDD(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(a.device)
    sd = torch.load(a.ckpt, map_location=a.device, weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    print(f"[dino-parcel] loaded {a.ckpt} ({time.time()-t0:.0f}s) | dist-peak watershed delineation", flush=True)

    manf = Path(a.data_dir) / "manifest.json"
    mm = json.loads(manf.read_text()) if manf.exists() else {}
    if a.prefix:                                                   # whole county / region
        te = sorted(p.stem for p in Path(a.data_dir).glob(f"{a.prefix}*.npz"))[:a.n_cells]
    elif "test" in mm:
        te = [n for n in mm["test"] if (Path(a.data_dir) / f"{n}.npz").exists()][:a.n_cells]
    else:                                                          # no test split (e.g. Changzhi) -> all cells
        te = sorted(p.stem for p in Path(a.data_dir).glob("*.npz"))[:a.n_cells]
    cache = {}
    g_single = None
    if a.dltb_parquet:                                             # one DLTB parquet for the whole region (cross-domain)
        import geopandas as gpd
        g_single = gpd.read_parquet(a.dltb_parquet)
        if g_single.crs is None or g_single.crs.to_epsg() != 4326:
            g_single = g_single.to_crs("EPSG:4326")
        try:
            g_single["geometry"] = g_single.geometry.make_valid()
        except Exception:
            g_single["geometry"] = g_single.geometry.buffer(0)
        cid = g_single["DLBM"].astype(str).str[:2]
        g_single["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0").astype(int)
        g_single = g_single[(g_single["cid"] >= 1) & (g_single["cid"] <= 2)].reset_index(drop=True)
        print(f"[dino-parcel] single DLTB {a.dltb_parquet}: {len(g_single)} cropland polygons", flush=True)
    if a.viz:
        Path(a.viz).mkdir(parents=True, exist_ok=True)
    TP = FP = FN = 0; over_all = []; under_all = []; bf1s = []
    pred_area_m = true_area_m = 0.0; parcels = []; nc = 0
    for i, n in enumerate(te):
        z = np.load(Path(a.data_dir) / f"{n}.npz"); x6 = z["x6"]; bbox = z["bbox"]
        _, H, W = x6.shape
        pix_m = (float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(
            math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W
        if g_single is not None:
            g = g_single
        else:
            try:
                g = load_county(a.dltb, n.split("_")[0], cache)
            except Exception as ex:
                print(f"  skip {n}: {ex}", flush=True); continue
        gt, _ = dltb_idmap(g, bbox, H, W)
        if gt is None:
            continue
        n_gt = int(gt.max())
        clsprob, dist, bnd = infer_heads(m, x6, a.device)
        pred, n_pred = dist_peak_instances(clsprob, dist, bnd, a.min_dist, a.peak_thr, a.min_area_px, a.ridge)
        tp, over, under = iou_match(pred, gt, n_pred, n_gt)
        TP += tp; FP += (n_pred - tp); FN += (n_gt - tp)
        over_all += over; under_all += under
        bf1s.append(bf1(edges(pred), edges(gt), 3))
        pred_area_m += float((pred > 0).sum()) * pix_m * pix_m
        true_area_m += float((gt > 0).sum()) * pix_m * pix_m
        for j in range(1, n_gt + 1):
            ga = int((gt == j).sum())
            if ga:
                parcels.append((ga * pix_m * pix_m, over[j - 1] if j - 1 < len(over) else 0))
        nc += 1
        if a.viz and i < 12:
            rgb = np.ascontiguousarray(x6[:3].transpose(1, 2, 0)).astype(np.uint8)
            _viz(rgb, gt, pred, Path(a.viz) / f"{n}.png")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(te)} | pred_inst≈{n_pred} gt={n_gt} ({time.time()-t0:.0f}s)", flush=True)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9), pr, rc
    F1, P, R = f1(TP, FP, FN)
    overm = float(np.mean([o for o in over_all if o > 0])) if over_all else 0
    underm = float(np.mean([u for u in under_all if u > 0])) if under_all else 0
    agree = 100.0 * min(pred_area_m, true_area_m) / (max(pred_area_m, true_area_m) + 1e-9)
    print(f"\n[dino-parcel] === {nc} cells | DINOv3-Sat dist-peak watershed vs DLTB cropland ===", flush=True)
    print(f"  INSTANCE-match F1 (IoU>=0.5) = {F1:.4f}  (P{P:.3f}/R{R:.3f})   TP{TP} FP{FP} FN{FN}", flush=True)
    print(f"  boundary-F1 (tol3px)         = {float(np.mean(bf1s)):.4f}", flush=True)
    print(f"  over-seg = {overm:.2f}   under-seg = {underm:.2f}", flush=True)
    print(f"  AREA-MATCH (FSDA口径) = {agree:.1f}%  (pred/ref={pred_area_m/(true_area_m+1e-9):.3f})", flush=True)
    out = {"ckpt": a.ckpt, "instance_f1": F1, "P": P, "R": R, "boundary_f1": float(np.mean(bf1s)),
           "over_seg": overm, "under_seg": underm, "area_agreement": agree, "n_cells": nc, "sweep": {}}
    print("  MMU(ha) recall sweep:", flush=True)
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


if __name__ == "__main__":
    main()
