"""榆中全县 单大图 统一 watershed + 帧场(Frame Field)累加 —— Stage A.
在 yz_global.py 基础上, band_worker 同一次前向额外取第4头帧场 o[3]=[B,4,H,W]
(c0_re,c0_im,c2_re,c2_im), /4 area-resize 乘 HANN4 累加到全局 /4 ff 累加器。
主进程 sum/cnt -> 全局 ffc0=ff[0]+1j*ff[1], ffc2=ff[2]+1j*ff[3]。
同时保存:
  - 全局 /4 idmap 栅格 (.npy, int32)  -> 供 Stage B 矢量化复用
  - cls_of (.json)                    -> 每 instance 的 class_id
  - 全局 /4 ffc0/ffc2 (.npy complex64) -> 供 Stage B 拓扑FFL
  - tr4 / crs / shape (.json)
  - shapes 矢量化的原始 region.parquet (保留旧产物口径, 对比用)
帧场 o[3] 经 Tanh, 与 ff_polygonize._tiled_ff 一致(那里用 raw norm6; 这里 train/eval 一致
用 enhance6 —— 帧场是方向场, enhance6 只调亮度对比, 主方向不变, 安全)。
"""
import argparse, json, os, sys, time, math
from pathlib import Path
import numpy as np

SIDECAR = "/home/ps/landform/sidecar"
MOSAIC  = "/mnt/sda/zf/landform/results/yuzhong_county_mosaic.tif"
WEIGHTS = "/mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt"
BACKBONE= "/home/ps/landform/dinov3/dinov3-vitl16-sat493m"
OUTDIR  = "/mnt/sda/zf/landform/results"
OUT_PARQUET = OUTDIR + "/yuzhong_globalffl_region.parquet"     # shapes 口径(对比), 非最终
ART = "/mnt/sda/zf/landform/results/yz_ffl_artifacts"          # 中间产物目录
CS = 448
STRIDE = 224
DS = 4
CS4 = CS // DS


def make_hann(cs):
    return np.maximum(np.outer(np.hanning(cs), np.hanning(cs)), 1e-3).astype(np.float32)


HANN4 = make_hann(CS4)


def band_worker(wid, gpu, row0, row1, H, W, ret_dict, log_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import rasterio
    from rasterio.windows import Window
    import cv2
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    from train_dino_1m_v3 import DinoV3FreqUNetBDDF, enhance6
    from train_dino_1m import norm6

    def log(msg):
        with open(log_path, "a") as f:
            f.write("[w%d gpu%d] %s\n" % (wid, gpu, msg)); f.flush()

    t0 = time.time()
    d3 = AutoModel.from_pretrained(BACKBONE, local_files_only=True)
    m = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).cuda()
    sd = torch.load(WEIGHTS, map_location="cuda", weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    log("model resident (%.0fs)" % (time.time() - t0))

    H4 = math.ceil(H / DS); W4 = math.ceil(W / DS)
    acc_cls = np.zeros((9, H4, W4), np.float32)
    acc_dist = np.zeros((H4, W4), np.float32)
    acc_bnd = np.zeros((H4, W4), np.float32)
    acc_ff = np.zeros((4, H4, W4), np.float32)            # c0_re,c0_im,c2_re,c2_im
    cnt = np.zeros((H4, W4), np.float32)

    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    xs_all = list(range(0, max(1, W - CS + 1), STRIDE))
    if xs_all[-1] != W - CS:
        xs_all.append(max(0, W - CS))
    ys = [t for t in ys_all if row0 <= t < row1]
    ndvi = np.zeros((5, CS, CS), np.float32)

    src = rasterio.open(MOSAIC)
    nwin = len(ys) * len(xs_all); done = 0
    log("band rows [%d,%d) -> %d y x %d x = %d windows" % (row0, row1, len(ys), len(xs_all), nwin))
    for t in ys:
        rowblk = src.read([1, 2, 3], window=Window(0, t, W, CS))
        for l in xs_all:
            tile = rowblk[:, :, l:l + CS]
            ch, cw = tile.shape[1], tile.shape[2]
            if ch < CS or cw < CS:
                pad = np.zeros((3, CS, CS), np.uint8); pad[:, :ch, :cw] = tile; tile = pad
            rgb = np.ascontiguousarray(tile).astype(np.uint8)
            x6 = np.concatenate([rgb, rgb], 0)
            xc = np.concatenate([norm6(enhance6(x6)), ndvi], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                o = m(xb); cl, bn, dsh, ff = o[0], o[1], o[2], o[3]
                if cl.shape[-2:] != (CS, CS):
                    cl = F.interpolate(cl, (CS, CS), mode="bilinear", align_corners=False)
                    bn = F.interpolate(bn, (CS, CS), mode="bilinear", align_corners=False)
                    dsh = F.interpolate(dsh, (CS, CS), mode="bilinear", align_corners=False)
                    ff = F.interpolate(ff, (CS, CS), mode="bilinear", align_corners=False)
                pr = torch.softmax(cl.float(), 1)[0].cpu().numpy()
                pd = torch.sigmoid(dsh.float())[0, 0].cpu().numpy()
                pb = torch.sigmoid(bn.float())[0, 0].cpu().numpy()
                pf = ff.float()[0].cpu().numpy()                    # (4,CS,CS) in [-1,1]
            pr4 = np.stack([cv2.resize(pr[c], (CS4, CS4), interpolation=cv2.INTER_AREA) for c in range(9)], 0)
            pd4 = cv2.resize(pd, (CS4, CS4), interpolation=cv2.INTER_AREA)
            pb4 = cv2.resize(pb, (CS4, CS4), interpolation=cv2.INTER_AREA)
            pf4 = np.stack([cv2.resize(pf[c], (CS4, CS4), interpolation=cv2.INTER_AREA) for c in range(4)], 0)
            t4 = t // DS; l4 = l // DS
            h4 = min(CS4, H4 - t4); w4 = min(CS4, W4 - l4)
            if h4 <= 0 or w4 <= 0:
                continue
            wn = HANN4[:h4, :w4]
            acc_cls[:, t4:t4 + h4, l4:l4 + w4] += pr4[:, :h4, :w4] * wn
            acc_dist[t4:t4 + h4, l4:l4 + w4] += pd4[:h4, :w4] * wn
            acc_bnd[t4:t4 + h4, l4:l4 + w4] += pb4[:h4, :w4] * wn
            acc_ff[:, t4:t4 + h4, l4:l4 + w4] += pf4[:, :h4, :w4] * wn
            cnt[t4:t4 + h4, l4:l4 + w4] += wn
            done += 1
        if (ys.index(t) + 1) % 5 == 0 or t == ys[-1]:
            r = (time.time() - t0) / max(done, 1)
            log("  %d/%d win | %.2fs/win | ETA %.0fmin" % (done, nwin, r, r * (nwin - done) / 60))
    src.close()
    shm = "/dev/shm/yz_globalffl_w%d.npz" % wid
    np.savez(shm, acc_cls=acc_cls, acc_dist=acc_dist, acc_bnd=acc_bnd, acc_ff=acc_ff, cnt=cnt)
    ret_dict[wid] = shm
    log("DONE %d windows -> %s (%.0fmin)" % (done, shm, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--out", default=OUT_PARQUET)
    ap.add_argument("--log", default="/tmp/yz_globalffl.log")
    a = ap.parse_args()
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import rasterio
    import torch.multiprocessing as mp

    os.makedirs(ART, exist_ok=True)
    src = rasterio.open(MOSAIC)
    H, W = src.height, src.width
    base_tr = src.transform; crs = src.crs
    src.close()
    H4 = math.ceil(H / DS); W4 = math.ceil(W / DS)
    from affine import Affine
    tr4 = base_tr * Affine.scale(DS, DS)
    print("[gffl] mosaic %dx%d -> /4 grid %dx%d | tr4=%s" % (H, W, H4, W4, tr4), flush=True)

    gpus = [int(g) for g in a.gpus.split(",")]; ng = len(gpus)
    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    per = math.ceil(len(ys_all) / ng)
    bands = []
    for i in range(ng):
        yi = ys_all[i * per:(i + 1) * per]
        if yi:
            bands.append((yi[0], yi[-1] + 1))
    print("[gffl] %d row-windows -> %d bands: %s" % (len(ys_all), len(bands), bands), flush=True)
    open(a.log, "w").write("[gffl] start %dx%d /4 %dx%d bands=%s\n" % (H, W, H4, W4, bands))

    ctx = mp.get_context("spawn"); mgr = ctx.Manager(); ret = mgr.dict()
    procs = []; t0 = time.time()
    for wid, (r0, r1) in enumerate(bands):
        p = ctx.Process(target=band_worker, args=(wid, gpus[wid % ng], r0, r1, H, W, ret, a.log))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    nb = len(bands)
    print("[gffl] all bands done (%.0fmin) -> merge" % ((time.time() - t0) / 60), flush=True)
    if len(ret) != nb:
        print("[gffl] FAIL: only %d/%d bands returned" % (len(ret), nb), flush=True)
        for p in procs:
            if p.exitcode not in (0, None):
                print("  proc exitcode", p.exitcode, flush=True)
        sys.exit(1)

    acc_cls = np.zeros((9, H4, W4), np.float32)
    acc_dist = np.zeros((H4, W4), np.float32)
    acc_bnd = np.zeros((H4, W4), np.float32)
    acc_ff = np.zeros((4, H4, W4), np.float32)
    cnt = np.zeros((H4, W4), np.float32)
    for wid in range(nb):
        z = np.load(ret[wid])
        acc_cls += z["acc_cls"]; acc_dist += z["acc_dist"]; acc_bnd += z["acc_bnd"]
        acc_ff += z["acc_ff"]; cnt += z["cnt"]; del z; os.remove(ret[wid])
    cnt = np.maximum(cnt, 1e-6)
    cls = acc_cls / cnt; dist = acc_dist / cnt; bnd = acc_bnd / cnt; ff = acc_ff / cnt
    del acc_cls, acc_dist, acc_bnd, acc_ff
    cov = float((cnt > 1e-3).mean())
    print("[gffl] merged. /4 cov=%.4f cls%s ff%s (%.0fmin)" % (cov, cls.shape, ff.shape, (time.time() - t0) / 60), flush=True)

    # 全局 watershed
    from dino_parcel_export import build_idmap, NAME_ZH, NAME_EN, HEX

    class Pp:
        min_dist = 20; peak_thr = 0.4; min_area_px = 200
        ridge = True; downscale = 1; smooth_iters = 1
    t1 = time.time()
    idmap, cls_of = build_idmap(cls, dist, bnd, Pp())
    del cls, dist, bnd
    nparc = len(cls_of)
    print("[gffl] build_idmap: %d instances (%.0fmin)" % (nparc, (time.time() - t1) / 60), flush=True)

    # ---- 保存中间产物供 Stage B 复用 ----
    np.save(ART + "/idmap4.npy", idmap.astype(np.int32))
    ffc0 = (ff[0] + 1j * ff[1]).astype(np.complex64)
    ffc2 = (ff[2] + 1j * ff[3]).astype(np.complex64)
    np.save(ART + "/ffc0.npy", ffc0); np.save(ART + "/ffc2.npy", ffc2)
    json.dump({str(k): int(v) for k, v in cls_of.items()}, open(ART + "/cls_of.json", "w"))
    json.dump({"H4": H4, "W4": W4, "tr4": list(tr4)[:6], "crs": str(crs)}, open(ART + "/meta.json", "w"))
    print("[gffl] artifacts saved -> %s (idmap4/ffc0/ffc2/cls_of/meta)" % ART, flush=True)

    # ---- shapes 矢量化(对比口径, 不平滑/不FFL) ----
    import rasterio.features
    from shapely.geometry import shape as _shape
    import geopandas as gpd
    t2 = time.time(); rows = []
    for geom, val in rasterio.features.shapes(idmap.astype(np.int32), mask=idmap > 0,
                                              connectivity=8, transform=tr4):
        c = cls_of.get(int(val))
        if not c:
            continue
        g = _shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            rows.append({"parcel_id": int(val), "class_id": c, "geometry": g})
        else:
            for pp in getattr(g, "geoms", []):
                if getattr(pp, "geom_type", "") == "Polygon" and not pp.is_empty and pp.area > 0:
                    rows.append({"parcel_id": int(val), "class_id": c, "geometry": pp})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf["area_m2"] = gdf.to_crs("EPSG:32648").geometry.area.round(1).values
    gdf["label"] = [NAME_ZH[c] for c in gdf["class_id"]]
    gdf["label_en"] = [NAME_EN[c] for c in gdf["class_id"]]
    gdf["rgb_hex"] = [HEX[c] for c in gdf["class_id"]]
    gdf = gdf.to_crs("EPSG:4326")
    gdf.insert(0, "gid", range(1, len(gdf) + 1))
    gdf.to_parquet(a.out)
    from collections import Counter
    cc = Counter(gdf["label"]); ar = gdf.groupby("label")["area_m2"].sum().div(1e6).round(1)
    print("[gffl] shapes vectorize: %d polys -> %s (%.0fmin)" % (len(gdf), a.out, (time.time() - t2) / 60), flush=True)
    print("  counts: %s" % dict(cc), flush=True)
    print("  km2: %s" % ar.to_dict(), flush=True)
    print("  TOTAL %.0fmin" % ((time.time() - t0) / 60), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
