"""test_memmap_infer — 验证 /N 累加器 band-local + 磁盘 memmap 与生产 infer_global 数值一致(parity)+ 省内存。

band-local: 每 worker 只分配**自己行带**的 /N 切片(非全图 H4)→ worker 内存 ~1/nbands。
主进程用**磁盘 memmap** 装配(带间 Hann 重叠行 += 求和)→ 主进程 RAM 不持全图。
不改生产 parcel_pipeline.py;此处对照生产 infer_global 做 parity,过了再考虑并入。
"""
import os, sys, math, time
sys.path.insert(0, "/home/ps/landform/sidecar")
import numpy as np
import parcel_pipeline as pp

CS, STRIDE = pp.CS, pp.STRIDE


def _accumulate_band_local(m, device, row0, row1, H, W, mosaic, ds):
    """同 pp._accumulate_band,但只分配本带 /N 切片。返回 (cls_loc, dist_loc, bnd_loc, cnt_loc, r0_local, done)。"""
    import rasterio
    from rasterio.windows import Window
    import cv2, torch
    import torch.nn.functional as F
    from train_dino_1m_v3 import enhance6
    from train_dino_1m import norm6
    is_cuda = str(device).startswith("cuda")
    cs4 = CS // ds
    hann4 = pp.make_hann(cs4)
    H4 = math.ceil(H / ds); W4 = math.ceil(W / ds)
    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    xs_all = list(range(0, max(1, W - CS + 1), STRIDE))
    if xs_all[-1] != W - CS:
        xs_all.append(max(0, W - CS))
    ys = [t for t in ys_all if row0 <= t < row1]
    if not ys:
        z = np.zeros
        return z((9, 0, W4), np.float32), z((0, W4), np.float32), z((0, W4), np.float32), z((0, W4), np.float32), 0, 0
    r0_local = ys[0] // ds
    r1_local = min(H4, ys[-1] // ds + cs4)
    hL = r1_local - r0_local
    acc_cls = np.zeros((9, hL, W4), np.float32)
    acc_dist = np.zeros((hL, W4), np.float32)
    acc_bnd = np.zeros((hL, W4), np.float32)
    cnt = np.zeros((hL, W4), np.float32)
    ndvi = np.zeros((5, CS, CS), np.float32)
    src = rasterio.open(mosaic); done = 0
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
            xb = torch.from_numpy(xc).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_cuda:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        o = m(xb)
                else:
                    o = m(xb)
                cl, bn, dsh = o[0], o[1], o[2]
                if cl.shape[-2:] != (CS, CS):
                    cl = F.interpolate(cl, (CS, CS), mode="bilinear", align_corners=False)
                    bn = F.interpolate(bn, (CS, CS), mode="bilinear", align_corners=False)
                    dsh = F.interpolate(dsh, (CS, CS), mode="bilinear", align_corners=False)
                pr = torch.softmax(cl.float(), 1)[0].cpu().numpy()
                pd = torch.sigmoid(dsh.float())[0, 0].cpu().numpy()
                pb = torch.sigmoid(bn.float())[0, 0].cpu().numpy()
            pr4 = np.stack([cv2.resize(pr[c], (cs4, cs4), interpolation=cv2.INTER_AREA) for c in range(9)], 0)
            pd4 = cv2.resize(pd, (cs4, cs4), interpolation=cv2.INTER_AREA)
            pb4 = cv2.resize(pb, (cs4, cs4), interpolation=cv2.INTER_AREA)
            t4 = t // ds; l4 = l // ds
            h4 = min(cs4, H4 - t4); w4 = min(cs4, W4 - l4)
            if h4 <= 0 or w4 <= 0:
                continue
            wn = hann4[:h4, :w4]
            rs = t4 - r0_local
            acc_cls[:, rs:rs + h4, l4:l4 + w4] += pr4[:, :h4, :w4] * wn
            acc_dist[rs:rs + h4, l4:l4 + w4] += pd4[:h4, :w4] * wn
            acc_bnd[rs:rs + h4, l4:l4 + w4] += pb4[:h4, :w4] * wn
            cnt[rs:rs + h4, l4:l4 + w4] += wn
            done += 1
    src.close()
    return acc_cls, acc_dist, acc_bnd, cnt, r0_local, done


def _band_worker_local(wid, gpu, row0, row1, H, W, mosaic, weights, backbone, ds, ret_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if "/home/ps/landform/sidecar" not in sys.path:
        sys.path.insert(0, "/home/ps/landform/sidecar")
    m = pp._load_model(backbone, weights, "cuda")
    cls, dist, bnd, cnt, r0, done = _accumulate_band_local(m, "cuda", row0, row1, H, W, mosaic, ds)
    f = "/tmp/memtest_w%d.npz" % wid
    np.savez(f, cls=cls, dist=dist, bnd=bnd, cnt=cnt, r0=r0)
    ret_dict[wid] = f


def infer_global_memmap(mosaic, weights, backbone, ds, gpus):
    import torch.multiprocessing as mp
    H, W, H4, W4, tr4, crs = pp._read_grid_meta(mosaic, ds)
    ng = len(gpus)
    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    per = math.ceil(len(ys_all) / ng)
    bands = []
    for i in range(ng):
        yi = ys_all[i * per:(i + 1) * per]
        if yi:
            bands.append((yi[0], yi[-1] + 1))
    ctx = mp.get_context("spawn"); mgr = ctx.Manager(); ret = mgr.dict()
    procs = []
    for wid, (r0, r1) in enumerate(bands):
        p = ctx.Process(target=_band_worker_local, args=(wid, gpus[wid % ng], r0, r1, H, W, mosaic, weights, backbone, ds, ret))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    md = "/tmp/memtest_mm"; os.makedirs(md, exist_ok=True)
    mm_cls = np.memmap(md + "/cls.dat", dtype=np.float32, mode="w+", shape=(9, H4, W4))
    mm_dist = np.memmap(md + "/dist.dat", dtype=np.float32, mode="w+", shape=(H4, W4))
    mm_bnd = np.memmap(md + "/bnd.dat", dtype=np.float32, mode="w+", shape=(H4, W4))
    mm_cnt = np.memmap(md + "/cnt.dat", dtype=np.float32, mode="w+", shape=(H4, W4))
    mm_cls[:] = 0; mm_dist[:] = 0; mm_bnd[:] = 0; mm_cnt[:] = 0
    for wid in range(len(bands)):
        z = np.load(ret[wid]); r0 = int(z["r0"]); hL = z["cls"].shape[1]
        mm_cls[:, r0:r0 + hL, :] += z["cls"]; mm_dist[r0:r0 + hL, :] += z["dist"]
        mm_bnd[r0:r0 + hL, :] += z["bnd"]; mm_cnt[r0:r0 + hL, :] += z["cnt"]
        del z; os.remove(ret[wid])
    cnt = np.maximum(np.asarray(mm_cnt), 1e-6)
    cls = np.asarray(mm_cls) / cnt; dist = np.asarray(mm_dist) / cnt; bnd = np.asarray(mm_bnd) / cnt
    return cls, dist, bnd, tr4, crs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--ds", type=int, default=4)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    t0 = time.time()
    print("[old] 生产 infer_global(全图累加器/worker)...", flush=True)
    co, disto, bndo, _, _, _ = pp.infer_global(a.mosaic, a.weights, a.backbone, a.ds, gpus)
    print("[old] done %.0fs cls%s" % (time.time() - t0, co.shape), flush=True)
    t1 = time.time()
    print("[new] infer_global_memmap(band-local + 磁盘 memmap)...", flush=True)
    cn, distn, bndn, _, _ = infer_global_memmap(a.mosaic, a.weights, a.backbone, a.ds, gpus)
    print("[new] done %.0fs cls%s" % (time.time() - t1, cn.shape), flush=True)
    dcls = float(np.abs(co - cn).max()); dd = float(np.abs(disto - distn).max()); db = float(np.abs(bndo - bndn).max())
    print("[parity] cls max|Δ|=%.3e | dist=%.3e | bnd=%.3e" % (dcls, dd, db), flush=True)
    H4, W4 = co.shape[1], co.shape[2]
    full_gb = (9 * H4 * W4 * 4 + 3 * H4 * W4 * 4) / 1e9
    print("[mem] 全图累加器/worker=%.2f GB(生产,每 worker 都持全图); band-local≈%.2f GB/worker(%d 带)" % (full_gb, full_gb / len(gpus), len(gpus)), flush=True)
    print("PARITY_PASS" if (dcls < 1e-4 and dd < 1e-4 and db < 1e-4) else "PARITY_FAIL", flush=True)


if __name__ == "__main__":
    main()
