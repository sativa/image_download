"""Does sub-meter (z18, ~0.4 m/px) imagery beat z17 (~0.8 m/px) at the parcel level?

Same cells, same 6ch model, two resolutions -> isolates the resolution variable. z17 input from
c_1m npz; z18 input from freshly downloaded esri+google GeoTIFFs (stacked -> 6ch). Ground truth =
Gansu DLTB polygons (resolution-agnostic). Reports parcel count-/area-F1 + MMU sweep for z17 vs z18
on the SAME test cells, so any gain is attributable to resolution alone (zero-shot: model trained on z17).
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2
from parcel_eval import load_county


@torch.no_grad()
def prob1m(model, x6, dev, cs=448):
    _, SZ, SZw = x6.shape
    acc = np.zeros((SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xb = torch.from_numpy(norm6(x6[:, t:t + cs, l:l + cs])).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb); lg = lg[0] if isinstance(lg, tuple) else lg
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0, 1].cpu().numpy()
            acc[t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
    return acc / np.maximum(cnt, 1)


def read_z18_x6(z18dir, name):
    """esri+google z18 GeoTIFFs -> (6,H,W) uint8 on the esri grid."""
    ep = Path(z18dir) / f"{name}_esri.tif"; gp = Path(z18dir) / f"{name}_google.tif"
    if not ep.exists() or not gp.exists():
        return None
    e = rasterio.open(ep).read()[:3]                      # (3,H,W)
    H, W = e.shape[1:]
    g = rasterio.open(gp).read()[:3]
    if g.shape[1:] != (H, W):
        g = F.interpolate(torch.from_numpy(g.astype(np.float32))[None], size=(H, W),
                          mode="bilinear", align_corners=False)[0].numpy()
    return np.concatenate([e, g], 0).astype(np.uint8)


def parcel_stats(prob, bbox, g, pix_m, min_px=50):
    """rasterize DLTB to prob grid; per-parcel majority vote -> list of (area_m2,true,pred)."""
    H, W = prob.shape
    tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
    idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
    if not idx:
        return []
    cb = shp_box(*bbox); sub = g.iloc[idx].reset_index(drop=True)
    sub = sub.iloc[[j for j in range(len(sub)) if sub.geometry.iloc[j].intersects(cb)]].reset_index(drop=True)
    if not len(sub):
        return []
    pid = rasterize([(sub.geometry.iloc[j], j + 1) for j in range(len(sub))],
                    out_shape=(H, W), transform=tr, fill=0, dtype="int32")
    out = []
    for j in range(len(sub)):
        m = pid == (j + 1); a = int(m.sum())
        if a < min_px:
            continue
        out.append((a * pix_m * pix_m, int(sub["cid"].iloc[j]) in (1, 2), bool(prob[m].mean() >= 0.5)))
    return out


def report(tag, parcels):
    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9)
    print(f"\n[{tag}] {len(parcels)} parcels", flush=True)
    print(f"  MMU(ha)  n      count-F1   area-F1", flush=True)
    for ha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        thr = ha * 10000
        ct = cfp = cfn = 0; at = afp = afn = 0.0; nn = 0
        for area, tc, pc in parcels:
            if area < thr:
                continue
            nn += 1
            if pc and tc: ct += 1; at += area
            elif pc and not tc: cfp += 1; afp += area
            elif not pc and tc: cfn += 1; afn += area
        if nn:
            print(f"  >={ha:<4}  {nn:>5d}   {f1(ct,cfp,cfn):.3f}      {f1(at,afp,afn):.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m_v2_bnd/best.pt")
    p.add_argument("--wrapper", action="store_true", default=True)
    p.add_argument("--regions", default="/mnt/sda/zf/landform/data/z18_test_regions.json")
    p.add_argument("--z18-dir", default="/mnt/sda/zf/landform/data/z18_test")
    p.add_argument("--c1m-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    base = DinoUNet5ch(dino, num_classes=3, in_channels=6, unfreeze_last_n=4)
    if a.wrapper:
        from train_dino_1m_v2 import DinoUNetBoundary
        model = DinoUNetBoundary(base)
    else:
        model = base
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()
                           and model.state_dict()[k].shape == v.shape})
    model = model.to(dev).eval()

    R = json.loads(Path(a.regions).read_text())
    cache = {}; pz17 = []; pz18 = []
    for r in R:
        name = f'{r["county"]}_{r["idx"]}'; bbox = r["bbox"]
        pix_w = (bbox[2] - bbox[0]) * 111320 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
        try:
            g = load_county(a.dltb, r["county"], cache)
        except Exception as ex:
            print(f"  skip {name}: {ex}", flush=True); continue
        # z17
        f17 = Path(a.c1m_dir) / f"{name}.npz"
        if f17.exists():
            x6 = np.load(f17)["x6"]; pix_m = pix_w / x6.shape[2]
            pz17 += parcel_stats(prob1m(model, x6, dev), bbox, g, pix_m)
        # z18
        x6b = read_z18_x6(a.z18_dir, name)
        if x6b is not None:
            pix_m = pix_w / x6b.shape[2]
            pz18 += parcel_stats(prob1m(model, x6b, dev), bbox, g, pix_m)
        print(f"  {name}: z17 {len(pz17)} / z18 {len(pz18)} parcels", flush=True)
    report(f"z17 (~0.8 m/px)  ckpt={Path(a.ckpt).parent.name}", pz17)
    report(f"z18 (~0.4 m/px)  ckpt={Path(a.ckpt).parent.name}", pz18)
    print("\n[z18-test] same cells, same model, zero-shot resolution transfer. Compare z17 vs z18.", flush=True)


if __name__ == "__main__":
    main()
