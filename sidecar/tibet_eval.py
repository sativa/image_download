"""Cross-domain head-to-head: run OUR Gansu-trained cropland model on Tibet (Xizang) imagery and
compare to the FSDA published parcels (Wang et al. 2025) as reference. Tibet is fully out-of-domain
(highland barley, ~4000 m, alien to the Loess Plateau training set) — a hard generalization test.

Reference = FSDA cropland parcels rasterized per cell. Reports pixel cropland-F1, IoU, and the
FSDA-style area-matching ratio (predicted cropland area / FSDA reference area)."""
import warnings; warnings.filterwarnings("ignore")
import json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box as shp_box

SC = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar"; sys.path.insert(0, SC)
from train_dino_1m import norm6
from train_dino_1m_v3 import DinoV3FreqUNet
from transformers import AutoModel

BK = os.path.expanduser("~/D/cropland_dino/dinov3-vitl16-sat493m")
CKPT = os.path.expanduser("~/D/cropland_dino/cropland_gdlxff.pt")
TIF = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/z17_tibet")
SHP = "/Users/zhangfeng/Downloads/xizang_fsda/Xizang cropland datasets/ALL_Xizang_Albers.shp"
REG = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/tibet_regions.json"
dev = "mps"


@torch.no_grad()
def prob_of(model, x6, ndvi, cs=448):
    _, SZ, SZw = x6.shape
    ph = max(0, cs - SZ); pw = max(0, cs - SZw)
    if ph or pw:
        x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), "edge"); ndvi = np.pad(ndvi, ((0, 0), (0, ph), (0, pw)))
    _, PH, PW = x6.shape
    acc = np.zeros((PH, PW), np.float32); cnt = np.zeros((PH, PW), np.float32)
    ys = list(range(0, max(1, PH - cs + 1), cs)); xs = list(range(0, max(1, PW - cs + 1), cs))
    if ys[-1] != PH - cs: ys.append(PH - cs)
    if xs[-1] != PW - cs: xs.append(PW - cs)
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t + cs, l:l + cs]), ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            lg = model(xb); lg = lg[0] if isinstance(lg, tuple) else lg
            if lg.shape[-2:] != (cs, cs):
                lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
            acc[t:t + cs, l:l + cs] += torch.softmax(lg.float(), 1)[0, 1].cpu().numpy(); cnt[t:t + cs, l:l + cs] += 1
    return (acc / np.maximum(cnt, 1))[:SZ, :SZw]


def main():
    d3 = AutoModel.from_pretrained(BK, local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=3, in_channels=11, unfreeze_last_n=4).to(dev)
    sd = torch.load(CKPT, map_location=dev, weights_only=True); msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval(); print("model loaded (MPS)", flush=True)
    print("loading FSDA parcels -> EPSG:3857 ...", flush=True)
    fsda = gpd.read_file(SHP).to_crs("EPSG:3857"); sidx = fsda.sindex
    print("FSDA parcels:", len(fsda), flush=True)

    regions = json.load(open(REG))
    px_tp = px_fp = px_fn = 0; pred_px = ref_px = 0; n = 0
    for r in regions:
        nm = f"XZ_{r['idx']}"; ep = TIF / f"{nm}_esri.tif"; gp = TIF / f"{nm}_google.tif"
        if not ep.exists():
            continue
        with rasterio.open(ep) as s:
            e = s.read()[:3]; H, W = s.height, s.width; tr = s.transform; bnds = s.bounds
        g = rasterio.open(gp).read()[:3] if gp.exists() else e
        if g.shape[1:] != (H, W):
            g = F.interpolate(torch.from_numpy(g.astype(np.float32))[None], size=(H, W), mode="bilinear",
                              align_corners=False)[0].clamp(0, 255).numpy()
        x6 = np.concatenate([e, g], 0).astype(np.uint8); ndvi = np.zeros((5, H, W), np.float32)
        prob = prob_of(model, x6, ndvi); pred = prob >= 0.5
        idx = list(sidx.intersection((bnds.left, bnds.bottom, bnds.right, bnds.top)))
        ref = np.zeros((H, W), np.uint8)
        if idx:
            cb = shp_box(bnds.left, bnds.bottom, bnds.right, bnds.top); sub = fsda.iloc[idx]
            shapes = [(geom, 1) for geom in sub.geometry if geom.intersects(cb)]
            if shapes:
                ref = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype=np.uint8)
        rc = ref == 1
        px_tp += int((pred & rc).sum()); px_fp += int((pred & ~rc).sum()); px_fn += int((~pred & rc).sum())
        pred_px += int(pred.sum()); ref_px += int(rc.sum()); n += 1
        if n % 10 == 0:
            print(f"  {n} cells done", flush=True)

    pr = px_tp / (px_tp + px_fp + 1e-9); rc = px_tp / (px_tp + px_fn + 1e-9)
    f1 = 2 * pr * rc / (pr + rc + 1e-9); iou = px_tp / (px_tp + px_fp + px_fn + 1e-9)
    ratio = pred_px / (ref_px + 1e-9); agree = 100 * min(pred_px, ref_px) / (max(pred_px, ref_px) + 1e-9)
    print(f"\n=== 西藏跨域 vs FSDA 参考 ({n} cells) ===", flush=True)
    print(f"  pixel cropland-F1 = {f1:.4f}  (P{pr:.3f}/R{rc:.3f})", flush=True)
    print(f"  IoU = {iou:.4f}", flush=True)
    print(f"  AREA-MATCH pred/ref = {ratio:.3f} -> {agree:.1f}% (FSDA 报 89.8% vs UAV)", flush=True)


if __name__ == "__main__":
    main()
