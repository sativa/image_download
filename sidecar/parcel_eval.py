"""PARCEL-LEVEL cropland evaluation — the unit CLAUDE.md actually mandates ("在 1m、地块级评估").

Pixel-level 1m-F1 saturates ~0.866 because DLTB rasterized at 1m has multi-pixel boundary noise +
image/survey temporal mismatch -> the LABEL itself caps agreement ~0.87, no model can beat label noise.
But DLTB is a PARCEL land survey: the deployable product is "what class is this parcel". Scoring per
PARCEL (each DLTB polygon = one unit, predicted class = majority vote of model pixels inside it)
absorbs boundary-pixel noise and is the correct accuracy for a land-class product.

For each test cell: run the DINOv2-1m model -> cropland prob; rasterize each DLTB polygon to a parcel-id
map; per parcel, pred_cropland = mean(prob[parcel])>0.5, true_cropland = (first-level DLBM in {1,2}).
Reports BOTH pixel-F1 (reference) and parcel-F1 (count-weighted + area-weighted) + parcel OA. Honest:
prints both so the pixel<->parcel gap is explicit.
"""
import argparse, json, math, sys, time
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
from train_dino_1m_v2 import norm6, load_ndvi_full, DinoUNetBoundary
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2


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
    g["cid"] = np.where(cid.str.isdigit(), cid.replace("", "0"), "0")
    g["cid"] = g["cid"].astype(int)
    g = g[(g["cid"] >= 1) & (g["cid"] <= 12)].reset_index(drop=True)
    cache[county] = g
    return g


@torch.no_grad()
def cropland_prob(model, x6, dev, cs=448, ndvi=None, enhance=False):
    """Tiled 1m cropland probability over a full cell (classifier branch, softmax class 1)."""
    _, SZ, SZw = x6.shape
    acc = np.zeros((SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xcrop = x6[:, t:t + cs, l:l + cs]
            if enhance:
                from train_dino_1m_v3 import enhance6
                xcrop = enhance6(xcrop)
            xc = norm6(xcrop)
            if ndvi is not None:
                xc = np.concatenate([xc, ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb)
                lg = lg[0] if isinstance(lg, tuple) else lg
                if lg.shape[-2:] != (cs, cs):
                    lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(lg.float(), 1)[0, 1].cpu().numpy()   # P(cropland)
            acc[t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
    return acc / np.maximum(cnt, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_1m_v2_bnd_mt/best.pt")
    p.add_argument("--multitemporal", action="store_true", help="set if ckpt is an 11ch (NDVI) model")
    p.add_argument("--plain", action="store_true", help="ckpt is a plain DinoUNet5ch (no boundary wrapper)")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--dltb", default=str(HOME / "data/v11_dltb"))
    p.add_argument("--n-cells", type=int, default=120)
    p.add_argument("--min-parcel-px", type=int, default=50, help="ignore tiny slivers (rasterization noise)")
    p.add_argument("--smp-arch", default="", help="eval an smp baseline: unet|deeplabv3plus|segformer")
    p.add_argument("--encoder", default="efficientnet-b5")
    p.add_argument("--v3-backbone", default="", help="dir of DINOv3-sat weights -> eval a DinoV3UNet ckpt")
    p.add_argument("--v3-freq", action="store_true", help="ckpt is a DinoV3FreqUNet (FreqFusion decoder)")
    p.add_argument("--v3-dysample", action="store_true", help="ckpt is a DinoV3DySampleUNet")
    p.add_argument("--v3-pointrend", action="store_true", help="ckpt is a DinoV3PointRendUNet (PointRend-lite)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--enhance", action="store_true", help="apply 1m RS image enhancement (CLAHE+unsharp) before norm")
    a = p.parse_args()
    dev = a.device; t0 = time.time()
    in_ch = 11 if a.multitemporal else 6

    if a.smp_arch:
        import segmentation_models_pytorch as smp
        B = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "segformer": smp.Segformer}
        model = B[a.smp_arch](encoder_name=a.encoder, encoder_weights=None, in_channels=6, classes=3).to(dev)
    elif a.v3_backbone:
        from train_dino_1m_v3 import DinoV3UNet, DinoV3FreqUNet, DinoV3DySampleUNet, DinoV3PointRendUNet
        from transformers import AutoModel
        d3 = AutoModel.from_pretrained(a.v3_backbone, local_files_only=True)
        M = (DinoV3PointRendUNet if a.v3_pointrend else DinoV3FreqUNet if a.v3_freq
             else DinoV3DySampleUNet if a.v3_dysample else DinoV3UNet)
        model = M(d3, num_classes=3, in_channels=in_ch, unfreeze_last_n=4).to(dev)
    else:
        from transformers import AutoModel
        dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
        base = DinoUNet5ch(dinov2, num_classes=3, in_channels=in_ch, unfreeze_last_n=4)
        model = (base if a.plain else DinoUNetBoundary(base))
    model = model.to(dev)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=True)
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()
                           and model.state_dict()[k].shape == v.shape}, strict=False); model.eval()
    print(f"[parcel] loaded {a.ckpt} in_ch={in_ch} ({time.time()-t0:.0f}s)", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te = [n for n in man["test"] if (Path(a.data_dir) / f"{n}.npz").exists()][:a.n_cells]
    cache = {}
    px_tp = px_fp = px_fn = 0
    parcels = []  # (area_px, true_crop, pred_crop) -> swept over min-px thresholds at the end
    n_parcels = 0
    for i, n in enumerate(te):
        z = np.load(Path(a.data_dir) / f"{n}.npz"); x6 = z["x6"]; lbl = z["label"]; bbox = z["bbox"]
        _, H, W = x6.shape
        pix_m = (float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W
        ndvi = load_ndvi_full(n, H, W) if a.multitemporal else None
        if a.multitemporal and ndvi is None:                  # new cells (e.g. terrace) lack v33 NDVI ->
            ndvi = np.zeros((5, H, W), np.float32)             # zero-fill, exactly as training/full_eval does
        prob = cropland_prob(model, x6, dev, ndvi=ndvi, enhance=a.enhance)
        pred = prob >= 0.5
        # pixel-level (reference)
        v = lbl > 0; gt = (lbl == 1) & v; pi = pred & v
        px_tp += int((pi & gt).sum()); px_fp += int((pi & ~gt & v).sum()); px_fn += int((~pred & gt).sum())
        # parcel-level: rasterize each DLTB polygon to a unique id
        try:
            g = load_county(a.dltb, n.split("_")[0], cache)
        except Exception as ex:
            print(f"  skip {n}: {ex}", flush=True); continue
        tr = from_bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H)
        idx = list(g.sindex.intersection((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
        if not idx:
            continue
        cb = shp_box(*bbox); sub = g.iloc[idx].reset_index(drop=True)
        shapes = [(geom, j + 1) for j, geom in enumerate(sub.geometry) if geom.intersects(cb)]
        if not shapes:
            continue
        pid = rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        for j in range(len(sub)):
            m = pid == (j + 1)
            area = int(m.sum())
            if area < 1:
                continue
            true_crop = int(sub["cid"].iloc[j]) in (1, 2)
            pred_crop = bool(prob[m].mean() >= 0.5)              # majority vote of model prob in parcel
            parcels.append((area * pix_m * pix_m, true_crop, pred_crop)); n_parcels += 1   # area in m^2 (resolution-independent)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(te)} parcels={n_parcels} ({time.time()-t0:.0f}s)", flush=True)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        return 2 * pr * rc / (pr + rc + 1e-9), pr, rc
    pxf, pxp, pxr = f1(px_tp, px_fp, px_fn)
    print(f"\n[parcel] === {a.n_cells} test cells, {n_parcels} parcels total ===", flush=True)
    print(f"  PIXEL cropland-F1 = {pxf:.4f} (P{pxp:.3f}/R{pxr:.3f})   <- saturated proxy", flush=True)
    # FSDA-style AREA-MATCHING accuracy: predicted cropland area / reference (DLTB) cropland area.
    pred_area = sum(ar for ar, tc, pc in parcels if pc)
    true_area = sum(ar for ar, tc, pc in parcels if tc)
    ratio = pred_area / (true_area + 1e-9)
    agree = 100.0 * min(pred_area, true_area) / (max(pred_area, true_area) + 1e-9)
    print(f"  AREA-MATCH (FSDA口径) pred/ref = {ratio:.3f}  -> {agree:.1f}% area agreement "
          f"(FSDA Tibet: 89.8%)", flush=True)
    print(f"  MMU(ha)  n_parcels  count-F1(P/R)        area-F1(P/R)         count-OA", flush=True)
    out = {"pixel_f1": pxf, "ckpt": a.ckpt, "sweep": {}}
    for ha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        thr = ha * 10000                                          # m^2 (resolution-independent MMU)
        ct = cf_ = cfn = 0; cn = 0; at = af_ = afn = 0.0; ok = 0
        for area, tc, pc in parcels:
            if area < thr:
                continue
            cn += 1
            if pc and tc: ct += 1; at += area; ok += 1
            elif pc and not tc: cf_ += 1; af_ += area
            elif not pc and tc: cfn += 1; afn += area
            else: ok += 1
        if cn == 0:
            continue
        cF, cP, cR = f1(ct, cf_, cfn); aF, aP, aR = f1(at, af_, afn)
        print(f"  >={ha:<5} {cn:>8d}   {cF:.3f}(P{cP:.2f}/R{cR:.2f})   {aF:.3f}(P{aP:.2f}/R{aR:.2f})   {ok/cn:.3f}", flush=True)
        out["sweep"][ha] = {"n": cn, "count_f1": cF, "area_f1": aF, "count_oa": ok / cn}
    json.dump(out, open("/mnt/sda/zf/landform/results/parcel_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
