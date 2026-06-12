"""榆中全县 单大图 统一 watershed —— 真无缝、零重叠、田块级。
全县 1m mosaic (71770x101635=7.3Gpx) 太大无法整图 watershed -> 用 downscale=4 全局累加器:
  滑窗(448, stride224, 50%重叠)在 1m mosaic 上推理 -> 每窗 softmax(cls)/sigmoid(dist,bnd)
  -> 降采样 /4 (area) -> 乘 /4 Hann 累加到全局 /4 网格 -> 累加完一次 build_idmap(ridge,downscale=1)
  -> 全局 idmap(无块,无缝,partition) -> 矢量化 -> GeoParquet。
多 GPU: 行方向切 4 带, 每 GPU 一带, 带间留 >=1 窗重叠保证 Hann 连续; 每 worker 自己的 /4 累加器,
最后主进程求和(无锁,完全正确)。enhance6 per-tile (配 bddf_enh 权重必须, train/eval 一致)。"""
import argparse, json, os, sys, time, math
from pathlib import Path
import numpy as np

SIDECAR = "/home/ps/landform/sidecar"
MOSAIC  = "/mnt/sda/zf/landform/results/yuzhong_county_mosaic.tif"
WEIGHTS = "/mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt"
BACKBONE= "/home/ps/landform/dinov3/dinov3-vitl16-sat493m"
OUT_PARQUET = "/mnt/sda/zf/landform/results/yuzhong_global_region.parquet"
CS = 448
STRIDE = 224
DS = 4                                   # downscale 全局网格
CS4 = CS // DS                           # 112


def make_hann(cs):
    w = np.maximum(np.outer(np.hanning(cs), np.hanning(cs)), 1e-3).astype(np.float32)
    return w


# /4 Hann (对 /4 网格的窗权重). 用 cs4 的 hanning, 与 /4 输出尺寸匹配
HANN4 = make_hann(CS4)                    # (112,112)


def band_worker(wid, gpu, row0, row1, H, W, ret_dict, log_path):
    """处理 mosaic 行区间 [row0,row1) 内所有以 t (窗顶行) 落在该带的滑窗。
    各 worker 维护自己的 /4 累加器(尺寸=全局), 完成后返回(主进程 sum)。"""
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
            f.write("[w%d gpu%d] %s\n" % (wid, gpu, msg))
            f.flush()

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
    cnt = np.zeros((H4, W4), np.float32)

    # 滑窗顶行/左列列表 (全局), 但只处理 t in [row0,row1) 的行带 (含右下补齐窗)
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
        # 整行一次性读 (CS 高 x 全宽) 比逐窗读快; 但全宽 71770x448x3 = ~96MB, ok
        rowblk = src.read([1, 2, 3], window=Window(0, t, W, CS))   # (3,CS,W)
        rh = rowblk.shape[1]                                       # 边缘可能<CS
        for l in xs_all:
            tile = rowblk[:, :, l:l + CS]                          # (3,h,w)
            ch, cw = tile.shape[1], tile.shape[2]
            if ch < CS or cw < CS:                                 # 右/下边缘补齐到 CS
                pad = np.zeros((3, CS, CS), np.uint8)
                pad[:, :ch, :cw] = tile; tile = pad
            rgb = np.ascontiguousarray(tile).astype(np.uint8)
            x6 = np.concatenate([rgb, rgb], 0)                     # (6,CS,CS) Esri||Esri(无Google)
            xc = np.concatenate([norm6(enhance6(x6)), ndvi], 0)    # per-tile enhance6 (同训练口径) + 0 NDVI
            xb = torch.from_numpy(xc).unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                o = m(xb); cl, bn, dsh = o[0], o[1], o[2]
                if cl.shape[-2:] != (CS, CS):
                    cl = F.interpolate(cl, (CS, CS), mode="bilinear", align_corners=False)
                    bn = F.interpolate(bn, (CS, CS), mode="bilinear", align_corners=False)
                    dsh = F.interpolate(dsh, (CS, CS), mode="bilinear", align_corners=False)
                pr = torch.softmax(cl.float(), 1)[0].cpu().numpy()         # (9,CS,CS)
                pd = torch.sigmoid(dsh.float())[0, 0].cpu().numpy()        # (CS,CS)
                pb = torch.sigmoid(bn.float())[0, 0].cpu().numpy()         # (CS,CS)
            # 降采样 /4 (area 平均)
            pr4 = np.stack([cv2.resize(pr[c], (CS4, CS4), interpolation=cv2.INTER_AREA) for c in range(9)], 0)
            pd4 = cv2.resize(pd, (CS4, CS4), interpolation=cv2.INTER_AREA)
            pb4 = cv2.resize(pb, (CS4, CS4), interpolation=cv2.INTER_AREA)
            # /4 网格目标位置
            t4 = t // DS; l4 = l // DS
            h4 = min(CS4, H4 - t4); w4 = min(CS4, W4 - l4)
            if h4 <= 0 or w4 <= 0:
                continue
            wn = HANN4[:h4, :w4]
            acc_cls[:, t4:t4 + h4, l4:l4 + w4] += pr4[:, :h4, :w4] * wn
            acc_dist[t4:t4 + h4, l4:l4 + w4] += pd4[:h4, :w4] * wn
            acc_bnd[t4:t4 + h4, l4:l4 + w4] += pb4[:h4, :w4] * wn
            cnt[t4:t4 + h4, l4:l4 + w4] += wn
            done += 1
        if (ys.index(t) + 1) % 5 == 0 or t == ys[-1]:
            r = (time.time() - t0) / max(done, 1)
            log("  %d/%d win | %.2fs/win | ETA %.0fmin" % (done, nwin, r, r * (nwin - done) / 60))
    src.close()
    # 存到 /dev/shm 让主进程读 (避免跨进程大数组 pickle)
    shm = "/dev/shm/yz_global_w%d.npz" % wid
    np.savez(shm, acc_cls=acc_cls, acc_dist=acc_dist, acc_bnd=acc_bnd, cnt=cnt)
    ret_dict[wid] = shm
    log("DONE %d windows -> %s (%.0fmin)" % (done, shm, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--out", default=OUT_PARQUET)
    ap.add_argument("--log", default="/tmp/yz_global.log")
    a = ap.parse_args()
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import rasterio
    import torch.multiprocessing as mp

    src = rasterio.open(MOSAIC)
    H, W = src.height, src.width
    base_tr = src.transform
    crs = src.crs
    src.close()
    H4 = math.ceil(H / DS); W4 = math.ceil(W / DS)
    # /4 transform = mosaic transform 缩放 x4 (像素变大4倍, 原点不变)
    from affine import Affine
    tr4 = base_tr * Affine.scale(DS, DS)
    print("[global] mosaic %dx%d -> /4 grid %dx%d | tr4=%s" % (H, W, H4, W4, tr4), flush=True)

    gpus = [int(g) for g in a.gpus.split(",")]
    ng = len(gpus)
    # 行带切分: 每带 row 数, 带间不需重叠 —— 因为不同带写入同一全局 /4 累加器(各自副本),
    # 主进程相加; Hann 窗本身跨带在 sum 后连续(窗 t 属于唯一一个带, 但其覆盖行会进入相邻带的 /4 行,
    # 累加叠加在最终 sum 中是连续的). 为绝对安全, 带边界对齐 STRIDE.
    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    nwin_total = len(ys_all)
    per = math.ceil(nwin_total / ng)
    bands = []
    for i in range(ng):
        yi = ys_all[i * per:(i + 1) * per]
        if not yi:
            continue
        bands.append((yi[0], yi[-1] + 1))     # [row0, row1) 顶行区间
    print("[global] %d row-windows -> %d bands: %s" % (nwin_total, len(bands), bands), flush=True)
    open(a.log, "w").write("[global] start %dx%d /4 %dx%d bands=%s\n" % (H, W, H4, W4, bands))

    ctx = mp.get_context("spawn")
    mgr = ctx.Manager(); ret = mgr.dict()
    procs = []
    t0 = time.time()
    for wid, (r0, r1) in enumerate(bands):
        p = ctx.Process(target=band_worker, args=(wid, gpus[wid % ng], r0, r1, H, W, ret, a.log))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    nb = len(bands)
    print("[global] all bands done (%.0fmin) -> merge accumulators" % ((time.time() - t0) / 60), flush=True)
    if len(ret) != nb:
        print("[global] FAIL: only %d/%d bands returned" % (len(ret), nb), flush=True)
        for p in procs:
            if p.exitcode not in (0, None):
                print("  proc exitcode", p.exitcode, flush=True)
        sys.exit(1)

    # 合并各带 /4 累加器
    acc_cls = np.zeros((9, H4, W4), np.float32)
    acc_dist = np.zeros((H4, W4), np.float32)
    acc_bnd = np.zeros((H4, W4), np.float32)
    cnt = np.zeros((H4, W4), np.float32)
    for wid in range(nb):
        z = np.load(ret[wid])
        acc_cls += z["acc_cls"]; acc_dist += z["acc_dist"]; acc_bnd += z["acc_bnd"]; cnt += z["cnt"]
        del z
        os.remove(ret[wid])
    cnt = np.maximum(cnt, 1e-6)
    cls = acc_cls / cnt; dist = acc_dist / cnt; bnd = acc_bnd / cnt
    del acc_cls, acc_dist, acc_bnd
    cov = float((cnt > 1e-3).mean())
    print("[global] merged. /4 coverage=%.4f | cls %s dist %s bnd %s (%.0fmin)" % (
        cov, cls.shape, dist.shape, bnd.shape, (time.time() - t0) / 60), flush=True)

    # === 全局 watershed (一次, 田块级 ridge, downscale=1 因已在 /4 网格) ===
    from dino_parcel_export import build_idmap, NAME_ZH, NAME_EN, HEX

    class P:
        min_dist = 20; peak_thr = 0.4; min_area_px = 200
        ridge = True; downscale = 1; smooth_iters = 1
    t1 = time.time()
    idmap, cls_of = build_idmap(cls, dist, bnd, P())
    del cls, dist, bnd
    nparc = len(cls_of)
    print("[global] build_idmap done: %d instances (%.0fmin)" % (nparc, (time.time() - t1) / 60), flush=True)

    # === 矢量化 (在 /4 网格 + tr4) ===
    import rasterio.features
    from shapely.geometry import shape as _shape
    import geopandas as gpd
    t2 = time.time()
    rows = []
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
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)     # EPSG:3857
    gdf["area_m2"] = gdf.to_crs("EPSG:32648").geometry.area.round(1).values
    gdf["label"] = [NAME_ZH[c] for c in gdf["class_id"]]
    gdf["label_en"] = [NAME_EN[c] for c in gdf["class_id"]]
    gdf["rgb_hex"] = [HEX[c] for c in gdf["class_id"]]
    gdf = gdf.to_crs("EPSG:4326")
    gdf.insert(0, "gid", range(1, len(gdf) + 1))
    gdf.to_parquet(a.out)
    from collections import Counter
    cc = Counter(gdf["label"]); ar = gdf.groupby("label")["area_m2"].sum().div(1e6).round(1)
    print("[global] vectorize done: %d polys -> %s (%.0fmin)" % (len(gdf), a.out, (time.time() - t2) / 60), flush=True)
    print("  counts: %s" % dict(cc), flush=True)
    print("  km2: %s" % ar.to_dict(), flush=True)
    print("  total %.0fmin" % ((time.time() - t0) / 60), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
