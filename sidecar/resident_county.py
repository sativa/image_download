"""Fast full-county parcel vectorization — RESIDENT, PER-CELL, multi-GPU.

Fixes the three bottlenecks profiled on the slow mosaic pipeline (a single 2048^2 cell took 67s but
only 9.7s of that was on the GPU): (1) model loaded ONCE per worker; (2) ONE merged sliding-window pass
producing cls+dist+bnd+frame-field together (was infer_heads + a second redundant _tiled_ff pass);
(3) per-cell small images so polygonize_ff and the EDT gap-fill never hit the big-mosaic blowup
(polygonize_ff is itself now bbox-sliced — see ff_polygonize.py).

Cells fan across GPUs via a spawn Pool: each worker pins one card (CUDA_VISIBLE_DEVICES set before the
torch import) using an atomic counter for an even spread, model resident across all its cells. Per-cell
GeoParquet written immediately (resumable); a final pass concats into the county GeoParquet + per-class km^2.
Seams: cells tile without overlap so an edge-straddling parcel is cut in two — area totals unchanged, only
a hairline at borders and a slightly inflated parcel COUNT."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SIDECAR = "/home/ps/landform/sidecar"
_M = {}


def _init(gpus, weights, backbone, counter):
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    gpu = gpus[wid % len(gpus)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import sys
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import torch
    from transformers import AutoModel
    from train_dino_1m_v3 import DinoV3FreqUNetBDDF
    d3 = AutoModel.from_pretrained(backbone, local_files_only=True)
    model = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).cuda()
    sd = torch.load(weights, map_location="cuda", weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()
    _M["model"] = model
    _M["gpu"] = gpu
    print(f"[worker {wid}] model resident on GPU {gpu}", flush=True)


def _infer_all(model, x6, cs=448):
    """ONE sliding-window pass -> (cls 9xHxW, dist HxW, bnd HxW, c0 complex, c2 complex). Model returns
    (cls, bnd, dist, ff). Zero NDVI (RGB-only deploy)."""
    import torch
    import torch.nn.functional as F
    from train_dino_1m import norm6
    _, H, W = x6.shape
    ndvi = np.zeros((5, H, W), np.float32)
    acc = np.zeros((9, H, W), np.float32)
    accd = np.zeros((H, W), np.float32)
    accb = np.zeros((H, W), np.float32)
    accf = np.zeros((4, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    st = max(1, cs // 2)
    ys = list(range(0, max(1, H - cs + 1), st)); xs = list(range(0, max(1, W - cs + 1), st))
    if ys[-1] != H - cs: ys.append(max(0, H - cs))
    if xs[-1] != W - cs: xs.append(max(0, W - cs))
    win = np.maximum(np.outer(np.hanning(cs), np.hanning(cs)), 1e-3).astype(np.float32)
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t + cs, l:l + cs]), ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                o = model(xb)
                cl, bn, ds, ff = o[0], o[1], o[2], o[3]
                if cl.shape[-2:] != (cs, cs):
                    cl = F.interpolate(cl, (cs, cs), mode="bilinear", align_corners=False)
                    bn = F.interpolate(bn, (cs, cs), mode="bilinear", align_corners=False)
                    ds = F.interpolate(ds, (cs, cs), mode="bilinear", align_corners=False)
                    ff = F.interpolate(ff, (cs, cs), mode="bilinear", align_corners=False)
                pr = torch.softmax(cl.float(), 1)[0].cpu().numpy()
                pd = torch.sigmoid(ds.float())[0, 0].cpu().numpy()
                pb = torch.sigmoid(bn.float())[0, 0].cpu().numpy()
                pf = ff.float()[0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += pr * win
            accd[t:t + cs, l:l + cs] += pd * win
            accb[t:t + cs, l:l + cs] += pb * win
            accf[:, t:t + cs, l:l + cs] += pf * win
            cnt[t:t + cs, l:l + cs] += win
    cnt = np.maximum(cnt, 1e-6)
    accf /= cnt
    return acc / cnt, accd / cnt, accb / cnt, accf[0] + 1j * accf[1], accf[2] + 1j * accf[3]


class _P:
    min_dist = 20; peak_thr = 0.4; min_area_px = 200; ridge = False; downscale = 1; smooth_iters = 1


def _process(task):
    cell, tif_dir, out_dir = task
    import sys
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import math

    import geopandas as gpd
    from sam3_classify.infer import read_rgb_from_geotiff
    from dino_parcel_export import build_idmap, NAME_ZH, NAME_EN, HEX
    from ff_polygonize import polygonize_ff
    op = Path(out_dir) / f"{cell}.parquet"
    if op.exists():
        return (cell, -1, "skip")
    t0 = time.time()
    try:
        tif = Path(tif_dir) / f"{cell}_esri.tif"
        if not tif.exists():
            return (cell, -3, "no_tif")
        rgb, profile, bbox = read_rgb_from_geotiff(tif)
        H, W, _ = rgb.shape
        rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8)
        x6 = np.concatenate([rgb_chw, rgb_chw], 0)
        cls, dist, bnd, c0, c2 = _infer_all(_M["model"], x6)
        idmap, cls_of = build_idmap(cls, dist, bnd, _P())
        transform = profile["transform"]; crs = profile["crs"]
        pix_m = (float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(
            math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W
        areas = np.bincount(idmap.ravel())
        rows = polygonize_ff(idmap, cls_of, c0, c2, transform, simp_px=2.0)
        for r in rows:
            pid = r["parcel_id"]; c = r["class_id"]
            r.update(label=NAME_ZH[c], label_en=NAME_EN[c], rgb_hex=HEX[c], cell=cell,
                     area_m2=round(float(areas[pid]) * pix_m * pix_m, 1) if pid < len(areas) else 0.0)
        if rows:
            gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_crs("EPSG:4326").to_parquet(op)
        else:
            gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326").to_parquet(op)
        return (cell, len(rows), f"{time.time() - t0:.0f}s")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return (cell, -2, str(ex)[:200])


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="/tmp/yz_full_regions.json")
    ap.add_argument("--tif-dir", default="/mnt/sda/zf/landform/data/yz_full_tif")
    ap.add_argument("--out-dir", default="/mnt/sda/zf/landform/results/yz_cells")
    ap.add_argument("--weights", default="/mnt/sda/zf/landform/results/dino_v3_bddf/last.pt")
    ap.add_argument("--backbone", default="/home/ps/landform/dinov3/dinov3-vitl16-sat493m")
    ap.add_argument("--county-out", default="/mnt/sda/zf/landform/results/yuzhong_full_region.parquet")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cells = json.loads(Path(a.regions).read_text())
    names = [f"{c['county']}_{c['idx']}" for c in cells]
    if a.limit:
        names = names[:a.limit]
    tasks = [(n, a.tif_dir, str(out_dir)) for n in names]
    nproc = len(gpus) * a.per_gpu
    t0 = time.time()
    print(f"[resident] {len(tasks)} cells on {len(gpus)} GPUs x {a.per_gpu} = {nproc} workers", flush=True)
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    ok = done = bad = 0
    # NOTE: do NOT use `with ctx.Pool(...)` — its __exit__ joins workers, and a spawn worker that loaded
    # a CUDA model often cannot be reaped cleanly, leaving the main process stuck in do_wait so the concat
    # below never runs (observed on the full-county run). Drive the pool explicitly and terminate() it.
    pool = ctx.Pool(nproc, initializer=_init, initargs=(gpus, a.weights, a.backbone, counter))
    try:
        for cell, n, msg in pool.imap_unordered(_process, tasks):
            done += 1
            if n >= 0:
                ok += 1
            elif n in (-2, -3):
                bad += 1
                print(f"  [{done}/{len(tasks)}] {cell} FAIL: {msg}", flush=True)
            if done <= 8 or done % 25 == 0 or done == len(tasks):
                rate = (time.time() - t0) / max(done, 1)
                eta = rate * (len(tasks) - done)
                print(f"  [{done}/{len(tasks)}] ok={ok} bad={bad} last={cell}:{msg} | {rate:.1f}s/cell wall | ETA {eta/60:.0f}min", flush=True)
    finally:
        pool.terminate()                                           # SIGTERM workers; skip the join that hangs on CUDA
    import geopandas as gpd
    import pandas as pd
    gdfs = []
    for n in names:
        p = out_dir / f"{n}.parquet"
        if not p.exists():
            continue
        try:
            d = gpd.read_parquet(p)
            if len(d):
                gdfs.append(d)
        except Exception as ex:
            print(f"  concat skip {n}: {ex}", flush=True)
    if gdfs:
        reg = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        reg.insert(0, "gid", range(1, len(reg) + 1))
        reg.to_parquet(a.county_out)
        from collections import Counter
        cc = Counter(reg["label"])
        ar = reg.groupby("label")["area_m2"].sum().div(1e6).round(1)
        print(f"[resident] DONE ok={ok} bad={bad} | {len(reg)} parcels -> {a.county_out}", flush=True)
        print(f"  counts: {dict(cc)}", flush=True)
        print(f"  km2: {ar.to_dict()}", flush=True)
    print(f"  total {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)", flush=True)
    os._exit(0)                                                    # hard-exit: skip atexit/CUDA cleanup that can hang


if __name__ == "__main__":
    main()
