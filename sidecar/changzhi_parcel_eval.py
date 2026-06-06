"""CROSS-PROVINCE parcel-level cropland eval (Changzhi, Shanxi) — same protocol as Gansu parcel_eval.

Ground truth = 长治市_DLTB_WGS84.parquet (840k DLTB polygons, DLBM land-class codes, WGS84 — same
schema as Gansu v11_dltb). Imagery = c_1m_changzhi (1m RGB). 6ch RGB-only model: 1m texture is
domain-invariant (the 1m-primary thesis) and Changzhi has no v33 NDVI, so multitemporal is N/A.

Per cell: model -> 1m cropland prob; query DLTB polygons intersecting the cell bbox; rasterize to a
parcel-id map; each parcel -> majority vote (cropland if mean prob>=0.5), truth = DLBM[:2] in {01,02}.
Reports pixel F1 + parcel count-/area-F1 over a minimum-mapping-unit sweep, directly comparable to the
in-domain Gansu 0.929/0.917. THIS IS THE CROSS-PROVINCE HEADLINE (no one globally reports parcel 0.9 here).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--wrapper", action="store_true", help="ckpt is a DinoUNetBoundary (v2) model")
    p.add_argument("--cz-1m-dir", default="/mnt/sda/zf/landform/data/c_1m_changzhi")
    p.add_argument("--dltb", default="/mnt/sda/zf/landform/data/changzhi_DLTB_wgs84.parquet")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device; t0 = time.time()

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

    print(f"[xprov] loading Changzhi DLTB ({time.time()-t0:.0f}s)...", flush=True)
    g = gpd.read_parquet(a.dltb)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except Exception:
        g["geometry"] = g.geometry.buffer(0)
    cid = g["DLBM"].astype(str).str[:2]
    g["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0")
    g["cid"] = g["cid"].astype(int)
    g = g[(g["cid"] >= 1) & (g["cid"] <= 12)].reset_index(drop=True)
    _ = g.sindex
    print(f"[xprov] {len(g)} polygons indexed ({time.time()-t0:.0f}s)", flush=True)

    names = sorted(pth.stem for pth in Path(a.cz_1m_dir).glob("*.npz"))
    px_tp = px_fp = px_fn = 0
    parcels = []  # (area_m2, true_crop, pred_crop)
    for i, n in enumerate(names):
        z = np.load(Path(a.cz_1m_dir) / f"{n}.npz"); x6 = z["x6"]; bbox = z["bbox"]
        _, H, W = x6.shape
        prob = prob1m(model, x6, dev); pred = prob >= 0.5
        tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
        idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
        if not idx:
            continue
        cb = shp_box(*bbox); sub = g.iloc[idx].reset_index(drop=True)
        keep = [j for j in range(len(sub)) if sub.geometry.iloc[j].intersects(cb)]
        sub = sub.iloc[keep].reset_index(drop=True)
        if not len(sub):
            continue
        pid = rasterize([(sub.geometry.iloc[j], j + 1) for j in range(len(sub))],
                        out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        crop_tr = rasterize([(sub.geometry.iloc[j], 1) for j in range(len(sub)) if int(sub["cid"].iloc[j]) in (1, 2)],
                            out_shape=(H, W), transform=tr, fill=0, dtype="uint8") if (sub["cid"].isin([1, 2]).any()) else np.zeros((H, W), "uint8")
        valid = pid > 0; gt = (crop_tr == 1) & valid; pi = pred & valid
        px_tp += int((pi & gt).sum()); px_fp += int((pi & ~gt & valid).sum()); px_fn += int((~pred & gt).sum())
        for j in range(len(sub)):
            m = pid == (j + 1); area = int(m.sum())
            if area < 50:
                continue
            parcels.append((float(area), int(sub["cid"].iloc[j]) in (1, 2), bool(prob[m].mean() >= 0.5)))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(names)} parcels={len(parcels)} ({time.time()-t0:.0f}s)", flush=True)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9), pr, rc
    pxf, pxp, pxr = f1(px_tp, px_fp, px_fn)
    print(f"\n[xprov-parcel] {Path(a.ckpt).parent.name} | {len(parcels)} parcels (Changzhi DLTB vector)", flush=True)
    print(f"  PIXEL cropland-F1 = {pxf:.4f} (P{pxp:.3f}/R{pxr:.3f})", flush=True)
    print(f"  MMU(ha)  n_parcels  count-F1(P/R)        area-F1(P/R)", flush=True)
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
            cF, cP, cR = f1(ct, cfp, cfn); aF, aP, aR = f1(at, afp, afn)
            print(f"  >={ha:<4}   {nn:>7d}   {cF:.3f}(P{cP:.2f}/R{cR:.2f})   {aF:.3f}(P{aP:.2f}/R{aR:.2f})", flush=True)


if __name__ == "__main__":
    main()
