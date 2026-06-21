"""parcel_pipeline — region-agnostic 1m 地块矢量管线(模型 -> 无缝矢量成品).

把榆中专用链(yz_global_ffl + yz_smooth2 + yz_postproc)通用化成 **不绑任何县/类别/降采样** 的可复用管线。
任意 1m mosaic + 四头 BDDF 权重 -> 全县无缝 7 类地块 GeoParquet。

阶段(都复用现有算法函数, 不重写):
  1. 全局 /N watershed 累加器推理(从 yz_global_ffl 抽出, **去掉 FFL 帧场**, 保留 cls/dist/bnd)。
     多 GPU 行带并行, 每窗 softmax(cls)/sigmoid(dist,bnd) -> /N area-resize -> Hann 累加到全局累加器。
     -> dino_parcel_export.build_idmap(ridge watershed) -> 全局 idmap(无块无缝 partition)。
  2. 全县整体矢量化 + 平滑(从 yz_smooth2 抽出): shapes 出干净分区 ->
     coverage_simplify(tol) 全县去 /N 阶梯 -> 一次 topojson.Topology -> 每 arc Chaikin(端点固定) -> 无缝。
  3. 可选裁县界: boundary 给了 -> 真几何 intersection 裁; 否则跳过。
  4. postproc.run_postproc 标准收尾(sliver / gap-hole / tiny-hole / invalid / standardize):
     强化的 eliminate_slivers(宽度判据 w<1m 并入邻块, 治 Chaikin 悬空线段) +
     drop_tiny_holes(删 <10m² 退化微洞, 治微楔/小环) 默认参数即够狠, 县级输出自动干净。

为何这样设计(被否方向见 PIPELINE.md):
  - **不用 FFL 帧场**: 帧场正则使多边形过直 + 逐实例重叠, 改 dist/bnd ridge watershed + 拓扑保持(topojson)。
  - **全局累加器** 而非 per-cell / 分块: 分块切线留白缝; per-cell 慢且有 cell 缝。全局一张 /N 网格累加 -> 无缝。
  - **全县一次 topojson + Chaikin arc**: 共享边只存一份 arc, Chaikin 后两侧逐点一致 -> 零重叠无缝。

设备可移植(--device auto|cuda|mps|cpu):
  - cuda  : 多 GPU 行带并行(torch.multiprocessing spawn, 每带绑一张卡), bf16 autocast。
  - mps/cpu: 单进程整图滑窗推理(无 mp.spawn), fp32(MPS 不支持 bf16 autocast)。
  auto = cuda > mps > cpu。Mac(M3 Ultra)走 mps。

输入二选一(必给其一):
  --mosaic <tif>     : 预拼的 1m RGB EPSG:3857 GeoTIFF。
  --cells-dir <dir>  : 一目录 per-cell EPSG:3857 tif -> rasterio.merge 自带拼接成临时 mosaic 再跑。

CLI:
  python parcel_pipeline.py \
    (--mosaic <tif> | --cells-dir <dir>) --weights <pt> --backbone <dir> \
    --boundary <parquet|none> --out <parquet> --device auto \
    --downscale 4 --smooth-iters 2 --classes classes.json --gpus 0,1,2,3

classes.json: list[[id, label_zh, label_en, [r,g,b]], ...] (默认 = dino_parcel_export.CLASSES, 7 类)。
"""
from __future__ import annotations
import argparse, json, math, os, sys, tempfile, time
from pathlib import Path

import numpy as np

SIDECAR = str(Path(__file__).resolve().parent)
CS = 448          # window size
STRIDE = 224      # 50% overlap


def make_hann(cs):
    return np.maximum(np.outer(np.hanning(cs), np.hanning(cs)), 1e-3).astype(np.float32)


def resolve_device(device):
    """'auto'|'cuda'|'mps'|'cpu' -> 实际可用设备字符串。auto: cuda>mps>cpu。"""
    import torch
    d = (device or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但 torch.cuda 不可用")
    if d == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps 但 torch.backends.mps 不可用")
    return d


# ===========================================================================
# 自带拼接: cells-dir -> 临时 mosaic
# ===========================================================================
def mosaic_from_cells(cells_dir, out_tif):
    """一目录 per-cell EPSG:3857 tif -> rasterio.merge.merge 拼成单 mosaic GeoTIFF(out_tif)。
    只取 *.tif(排除 .landform./_mosaic 产品)。返回 out_tif。"""
    import rasterio
    from rasterio.merge import merge as rio_merge

    cells_dir = Path(cells_dir)
    tifs = sorted(p for p in cells_dir.glob("*.tif")
                  if ".landform" not in p.name and "_mosaic" not in p.name)
    if not tifs:
        raise RuntimeError("--cells-dir %s 下没有可拼接的 *.tif" % cells_dir)
    print("[mosaic] merging %d cell tifs from %s" % (len(tifs), cells_dir), flush=True)
    srcs = [rasterio.open(p) for p in tifs]
    try:
        mos, mos_tr = rio_merge(srcs)            # (bands, H, W), nodata-aware
        meta = srcs[0].meta.copy()
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    # 只保留前 3 波段(RGB); 4 波段 RGBA 输入丢 alpha。
    nb = min(3, mos.shape[0])
    mos = mos[:nb]
    meta.update(driver="GTiff", height=mos.shape[1], width=mos.shape[2],
                count=nb, transform=mos_tr, crs=crs, dtype=mos.dtype,
                compress="deflate", tiled=True,
                bigtiff="IF_SAFER")   # 文件可能 >4GB 时自动 BIGTIFF(地级市/省级 mosaic);县级照旧经典 TIFF
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(mos)
    print("[mosaic] wrote %s (%dx%d, %d bands, crs=%s)" %
          (out_tif, mos.shape[2], mos.shape[1], nb, crs), flush=True)
    return out_tif


# ===========================================================================
# 阶段 1: 全局 /N watershed 累加器推理(去 FFL, 保 cls/dist/bnd)
# ===========================================================================
def _load_model(backbone, weights, device):
    """加载四头 BDDF 模型到 device(device-agnostic)。返回 eval() 模型。"""
    import torch
    from transformers import AutoModel
    from train_dino_1m_v3 import DinoV3FreqUNetBDDF

    d3 = AutoModel.from_pretrained(backbone, local_files_only=True)
    m = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(device)
    sd = torch.load(weights, map_location=device, weights_only=True)
    msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    return m


def _accumulate_band(m, device, row0, row1, H, W, mosaic, ds, log=print):
    """对行带 [row0,row1) 做滑窗推理, 累加到 (acc_cls, acc_dist, acc_bnd, cnt) /N 网格。
    device-agnostic: cuda 用 bf16 autocast; mps/cpu 用 fp32(MPS 不支持 bf16 autocast)。
    返回这条带的 4 个累加数组(全局 /N 尺寸, 带外为 0)。"""
    import rasterio
    from rasterio.windows import Window
    import cv2
    import torch
    import torch.nn.functional as F
    from train_dino_1m_v3 import enhance6
    from train_dino_1m import norm6

    is_cuda = str(device).startswith("cuda")
    cs4 = CS // ds
    hann4 = make_hann(cs4)
    H4 = math.ceil(H / ds); W4 = math.ceil(W / ds)
    acc_cls = np.zeros((9, H4, W4), np.float32)
    acc_dist = np.zeros((H4, W4), np.float32)
    acc_bnd = np.zeros((H4, W4), np.float32)
    cnt = np.zeros((H4, W4), np.float32)

    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    xs_all = list(range(0, max(1, W - CS + 1), STRIDE))
    if xs_all[-1] != W - CS:
        xs_all.append(max(0, W - CS))
    ys = [t for t in ys_all if row0 <= t < row1]
    ndvi = np.zeros((5, CS, CS), np.float32)

    src = rasterio.open(mosaic)
    nwin = len(ys) * len(xs_all); done = 0; t0 = time.time()
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
            xb = torch.from_numpy(xc).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_cuda:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        o = m(xb)
                else:
                    o = m(xb)                       # mps/cpu: fp32, no autocast
                cl, bn, dsh = o[0], o[1], o[2]      # 忽略 o[3] 帧场
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
            acc_cls[:, t4:t4 + h4, l4:l4 + w4] += pr4[:, :h4, :w4] * wn
            acc_dist[t4:t4 + h4, l4:l4 + w4] += pd4[:h4, :w4] * wn
            acc_bnd[t4:t4 + h4, l4:l4 + w4] += pb4[:h4, :w4] * wn
            cnt[t4:t4 + h4, l4:l4 + w4] += wn
            done += 1
        if (ys.index(t) + 1) % 5 == 0 or t == ys[-1]:
            r = (time.time() - t0) / max(done, 1)
            log("  %d/%d win | %.2fs/win | ETA %.0fmin" % (done, nwin, r, r * (nwin - done) / 60))
    src.close()
    return acc_cls, acc_dist, acc_bnd, cnt, done


def _band_worker(wid, gpu, row0, row1, H, W, mosaic, weights, backbone, ds, ret_dict, log_path):
    """一条行带的滑窗推理 worker(在子进程内, 绑定一张 GPU)。CUDA-only 多GPU路径。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)

    def log(msg):
        with open(log_path, "a") as f:
            f.write("[w%d gpu%s] %s\n" % (wid, gpu, msg)); f.flush()

    t0 = time.time()
    m = _load_model(backbone, weights, "cuda")
    log("model resident (%.0fs)" % (time.time() - t0))
    acc_cls, acc_dist, acc_bnd, cnt, done = _accumulate_band(
        m, "cuda", row0, row1, H, W, mosaic, ds, log=log)
    # scratch: /dev/shm(RAM,快)放不下 npz 时落 /tmp(磁盘)——治大区域 /N 累加器写满 shm/爆内存。
    # 每 npz ≈ (9·H4·W4 + 3·H4·W4)·4B(长治地级市 /4 ≈80GB/worker);县级小,照旧走 /dev/shm。
    need = acc_cls.nbytes + acc_dist.nbytes + acc_bnd.nbytes + cnt.nbytes
    scratch = "/dev/shm"
    if not os.path.isdir(scratch):
        scratch = "/tmp"
    else:
        st = os.statvfs(scratch)
        if st.f_bavail * st.f_frsize < need * 1.1:
            scratch = "/tmp"
    shm = "%s/parcelpipe_w%d.npz" % (scratch, wid)
    np.savez(shm, acc_cls=acc_cls, acc_dist=acc_dist, acc_bnd=acc_bnd, cnt=cnt)
    ret_dict[wid] = shm
    log("DONE %d windows -> %s (%.0fmin)" % (done, shm, (time.time() - t0) / 60))


def _read_grid_meta(mosaic, ds):
    """读 mosaic 几何 -> (H, W, H4, W4, tr4, crs)。"""
    import rasterio
    from affine import Affine
    src = rasterio.open(mosaic)
    H, W = src.height, src.width
    base_tr = src.transform; crs = src.crs
    src.close()
    H4 = math.ceil(H / ds); W4 = math.ceil(W / ds)
    tr4 = base_tr * Affine.scale(ds, ds)
    return H, W, H4, W4, tr4, crs


def _finalize_acc(acc_cls, acc_dist, acc_bnd, cnt, ds, t0):
    cnt = np.maximum(cnt, 1e-6)
    cls = acc_cls / cnt; dist = acc_dist / cnt; bnd = acc_bnd / cnt
    cov = float((cnt > 1e-3).mean())
    print("[infer] merged. /%d cov=%.4f cls%s (%.0fmin)" %
          (ds, cov, cls.shape, (time.time() - t0) / 60), flush=True)
    return cls, dist, bnd, cov


def infer_single_device(mosaic, weights, backbone, ds, device, log_path="/tmp/parcel_pipeline_infer.log"):
    """单设备(mps/cpu, 或单 GPU)整图滑窗推理 -> (cls, dist, bnd, tr4, crs, cov)。
    单进程, 不 mp.spawn; 整图一条带(row0=0,row1=H)滑窗。"""
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    H, W, H4, W4, tr4, crs = _read_grid_meta(mosaic, ds)
    print("[infer] mosaic %dx%d -> /%d grid %dx%d | device=%s | tr4=%s" %
          (H, W, ds, H4, W4, device, tr4), flush=True)
    open(log_path, "w").write("[infer] single-device %s %dx%d /%d %dx%d\n" % (device, H, W, ds, H4, W4))
    t0 = time.time()
    m = _load_model(backbone, weights, device)
    print("[infer] model resident on %s (%.0fs)" % (device, time.time() - t0), flush=True)

    def log(msg):
        print("[infer] " + msg, flush=True)

    acc_cls, acc_dist, acc_bnd, cnt, _ = _accumulate_band(
        m, device, 0, H, H, W, mosaic, ds, log=log)
    cls, dist, bnd, cov = _finalize_acc(acc_cls, acc_dist, acc_bnd, cnt, ds, t0)
    return cls, dist, bnd, tr4, crs, cov


def infer_global(mosaic, weights, backbone, ds, gpus, log_path="/tmp/parcel_pipeline_infer.log"):
    """全局 /N 累加器推理(CUDA 多 GPU)-> (cls(9,H4,W4), dist, bnd, tr4, crs, cov)。

    多 GPU 行带并行(spawn), 每带一个 worker -> 主进程归并 sum/cnt。
    单 GPU 列表(如 ['0'])也支持 -> 单带串行。
    非 CUDA 设备请走 infer_single_device。
    """
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import torch.multiprocessing as mp

    H, W, H4, W4, tr4, crs = _read_grid_meta(mosaic, ds)
    print("[infer] mosaic %dx%d -> /%d grid %dx%d | tr4=%s" % (H, W, ds, H4, W4, tr4), flush=True)

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
    print("[infer] %d row-windows -> %d bands: %s" % (len(ys_all), len(bands), bands), flush=True)
    open(log_path, "w").write("[infer] start %dx%d /%d %dx%d bands=%s\n" % (H, W, ds, H4, W4, bands))

    ctx = mp.get_context("spawn"); mgr = ctx.Manager(); ret = mgr.dict()
    procs = []; t0 = time.time()
    for wid, (r0, r1) in enumerate(bands):
        p = ctx.Process(target=_band_worker,
                        args=(wid, gpus[wid % ng], r0, r1, H, W, mosaic, weights, backbone, ds, ret, log_path))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    nb = len(bands)
    if len(ret) != nb:
        raise RuntimeError("infer FAIL: only %d/%d bands returned (exitcodes=%s)" %
                           (len(ret), nb, [p.exitcode for p in procs]))

    acc_cls = np.zeros((9, H4, W4), np.float32)
    acc_dist = np.zeros((H4, W4), np.float32)
    acc_bnd = np.zeros((H4, W4), np.float32)
    cnt = np.zeros((H4, W4), np.float32)
    for wid in range(nb):
        z = np.load(ret[wid])
        acc_cls += z["acc_cls"]; acc_dist += z["acc_dist"]; acc_bnd += z["acc_bnd"]; cnt += z["cnt"]
        del z; os.remove(ret[wid])
    cls, dist, bnd, cov = _finalize_acc(acc_cls, acc_dist, acc_bnd, cnt, ds, t0)
    return cls, dist, bnd, tr4, crs, cov


# ===========================================================================
# band-local + 磁盘 memmap 全局推理(与 infer_global 数值逐位一致, test_memmap_infer 验证 max|Δ|=0)。
# 每 worker 只分配本带 /N 切片(内存 ~1/带数), 主进程用磁盘 memmap 装配(带间 Hann 重叠行 += 求和)。
# -> 根治"每 worker 全图累加器"内存墙(大市/省级满卡不爆)。cuda 多卡默认走此路(_run_from_mosaic)。
# ===========================================================================
def _accumulate_band_local(m, device, row0, row1, H, W, mosaic, ds, log=print):
    """同 _accumulate_band, 但只分配本带 /N 切片。返回 (cls_loc, dist_loc, bnd_loc, cnt_loc, r0_local, done)。"""
    import rasterio
    from rasterio.windows import Window
    import cv2
    import torch
    import torch.nn.functional as F
    from train_dino_1m_v3 import enhance6
    from train_dino_1m import norm6
    is_cuda = str(device).startswith("cuda")
    cs4 = CS // ds
    hann4 = make_hann(cs4)
    H4 = math.ceil(H / ds); W4 = math.ceil(W / ds)
    ys_all = list(range(0, max(1, H - CS + 1), STRIDE))
    if ys_all[-1] != H - CS:
        ys_all.append(max(0, H - CS))
    xs_all = list(range(0, max(1, W - CS + 1), STRIDE))
    if xs_all[-1] != W - CS:
        xs_all.append(max(0, W - CS))
    ys = [t for t in ys_all if row0 <= t < row1]
    if not ys:
        return (np.zeros((9, 0, W4), np.float32), np.zeros((0, W4), np.float32),
                np.zeros((0, W4), np.float32), np.zeros((0, W4), np.float32), 0, 0)
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


def _band_worker_local(wid, gpu, row0, row1, H, W, mosaic, weights, backbone, ds, ret_dict, scratch):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    m = _load_model(backbone, weights, "cuda")
    cls, dist, bnd, cnt, r0, done = _accumulate_band_local(m, "cuda", row0, row1, H, W, mosaic, ds)
    f = "%s/parcelpipe_local_w%d.npz" % (scratch, wid)
    np.savez(f, cls=cls, dist=dist, bnd=bnd, cnt=cnt, r0=r0)
    ret_dict[wid] = f


def infer_global_memmap(mosaic, weights, backbone, ds, gpus, log_path="/tmp/parcel_pipeline_infer.log"):
    """band-local + 磁盘 memmap 版 infer_global(数值逐位一致 / 内存 ~1/带数 / 主进程装配不持全图)。
    返回 (cls(9,H4,W4), dist, bnd, tr4, crs, cov)。slices/memmap 走磁盘 /tmp scratch(非 /dev/shm RAM)。"""
    import torch.multiprocessing as mp
    import tempfile, shutil
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    H, W, H4, W4, tr4, crs = _read_grid_meta(mosaic, ds)
    print("[infer] (memmap) mosaic %dx%d -> /%d grid %dx%d" % (H, W, ds, H4, W4), flush=True)
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
    print("[infer] (memmap) %d row-windows -> %d bands: %s" % (len(ys_all), len(bands), bands), flush=True)
    scratch = tempfile.mkdtemp(prefix="parcelpipe_mm_", dir="/tmp")
    ctx = mp.get_context("spawn"); mgr = ctx.Manager(); ret = mgr.dict()
    procs = []; t0 = time.time()
    for wid, (r0, r1) in enumerate(bands):
        p = ctx.Process(target=_band_worker_local,
                        args=(wid, gpus[wid % ng], r0, r1, H, W, mosaic, weights, backbone, ds, ret, scratch))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    if len(ret) != len(bands):
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError("infer(memmap) FAIL: %d/%d bands (exit=%s)" % (len(ret), len(bands), [p.exitcode for p in procs]))
    try:
        mm_cls = np.memmap(scratch + "/cls.dat", dtype=np.float32, mode="w+", shape=(9, H4, W4))
        mm_dist = np.memmap(scratch + "/dist.dat", dtype=np.float32, mode="w+", shape=(H4, W4))
        mm_bnd = np.memmap(scratch + "/bnd.dat", dtype=np.float32, mode="w+", shape=(H4, W4))
        mm_cnt = np.memmap(scratch + "/cnt.dat", dtype=np.float32, mode="w+", shape=(H4, W4))
        for wid in range(len(bands)):
            z = np.load(ret[wid]); r0 = int(z["r0"]); hL = z["cls"].shape[1]
            if hL:
                mm_cls[:, r0:r0 + hL, :] += z["cls"]; mm_dist[r0:r0 + hL, :] += z["dist"]
                mm_bnd[r0:r0 + hL, :] += z["bnd"]; mm_cnt[r0:r0 + hL, :] += z["cnt"]
            del z; os.remove(ret[wid])
        cnt = np.maximum(np.asarray(mm_cnt), 1e-6)
        cov = float((np.asarray(mm_cnt) > 1e-3).mean())
        cls = np.asarray(mm_cls) / cnt
        dist = np.asarray(mm_dist) / cnt
        bnd = np.asarray(mm_bnd) / cnt
        del mm_cls, mm_dist, mm_bnd, mm_cnt
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("[infer] (memmap) merged. /%d cov=%.4f cls%s (%.0fmin)" % (ds, cov, cls.shape, (time.time() - t0) / 60), flush=True)
    return cls, dist, bnd, tr4, crs, cov


def idmap_from_heads(cls, dist, bnd, min_dist=20, peak_thr=0.4, min_area_px=200, ridge=True):
    """全局 idmap: cropland/orchard 用 dist-peak ridge watershed, 其余类用 argmax 连通域。
    复用 dino_parcel_export.build_idmap(全局 /N 网格, downscale=1)。返回 (idmap, cls_of)。"""
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    from dino_parcel_export import build_idmap

    class Pp:
        pass
    Pp.min_dist = min_dist; Pp.peak_thr = peak_thr; Pp.min_area_px = min_area_px
    Pp.ridge = ridge; Pp.downscale = 1; Pp.smooth_iters = 1
    t0 = time.time()
    idmap, cls_of = build_idmap(cls, dist, bnd, Pp)
    print("[idmap] build_idmap: %d instances (%.0fmin)" % (len(cls_of), (time.time() - t0) / 60), flush=True)
    return idmap, cls_of


# ===========================================================================
# 阶段 2: 全县整体矢量化 + Chaikin 平滑(region-agnostic, 从 yz_smooth2 抽出)
# ===========================================================================
def chaikin_arc(arc, iters):
    """Chaikin corner-cutting on a single arc; FIRST & LAST point(=nodes) FIXED。
    arc: list of [x,y]。每段用 1/4、3/4 切角点替换, 迭代 iters 次; arc 端点(节点)不动 -> 共享边逐点一致 -> 无缝。"""
    pts = np.asarray(arc, dtype=np.float64)
    if iters <= 0 or len(pts) < 3:
        return arc
    p = pts
    for _ in range(iters):
        if len(p) < 3:
            break
        a = p[:-1]; b = p[1:]
        q = a + 0.25 * (b - a)
        r = a + 0.75 * (b - a)
        inner = np.empty((2 * len(q), 2), dtype=np.float64)
        inner[0::2] = q; inner[1::2] = r
        new = np.empty((len(inner) + 2, 2), dtype=np.float64)
        new[0] = p[0]; new[1:-1] = inner; new[-1] = p[-1]
        p = new
    return p.tolist()


def vectorize_idmap(idmap, cls_of, tr4, crs):
    """全局 idmap -> 干净分区 GeoDataFrame(class_id + geometry), CRS=源 CRS。
    用 rasterio.features.shapes 出多边形(无缝 coverage), 修无效, explode 单 Polygon。"""
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    import rasterio.features
    from shapely.geometry import shape as _shape
    import geopandas as gpd

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
            rows.append({"parcel_id": int(val), "class_id": int(c), "geometry": g})
        else:
            for pp in getattr(g, "geoms", []):
                if getattr(pp, "geom_type", "") == "Polygon" and not pp.is_empty and pp.area > 0:
                    rows.append({"parcel_id": int(val), "class_id": int(c), "geometry": pp})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gdf


def resolve_overlaps(gdf, work_crs="EPSG:3857", min_area=1e-4):
    """治 Chaikin 残留重叠 -> exact 零重叠(只动重叠对, 县级可扩展)。

    Chaikin 在被 topojson 留成两条单引用 arc 的极少数内部边上, 两份独立移动 -> 局部重叠(本测试 ~1%)。
    coverage_simplify(tol=0) snap 治不了真重叠。这里 STRtree 找重叠对, 重叠区从**较小**地块挖掉(归较大者),
    union 不变 -> 无新缝。只触碰重叠对(非全量), O(#overlaps), 县级可扩展。
    """
    import geopandas as gpd
    import shapely
    from shapely import make_valid

    geoms = list(gdf.geometry.values)
    arr = np.array(geoms, dtype=object)
    areas = shapely.area(arr)
    tree = shapely.STRtree(arr)
    pairs = tree.query(arr, predicate="overlaps")
    seen = set(); n_fixed = 0
    for a_i, b_i in zip(pairs[0], pairs[1]):
        i, j = int(a_i), int(b_i)
        if i >= j or (i, j) in seen:
            continue
        seen.add((i, j))
        try:
            inter = geoms[i].intersection(geoms[j])
        except Exception:
            continue
        if inter.is_empty or inter.area <= min_area:
            continue
        loser = i if areas[i] < areas[j] else j   # 较小者让出重叠区
        gm = geoms[loser].difference(inter)
        if not gm.is_valid:
            gm = make_valid(gm)
        geoms[loser] = gm; n_fixed += 1
    out = gpd.GeoDataFrame({"class_id": gdf["class_id"].values},
                           geometry=gpd.GeoSeries(geoms, crs=gdf.crs))
    out = out[~out.geometry.is_empty & out.geometry.notna()]
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    return out, n_fixed


def _iter_arc_ids(arcs):
    """递归遍历 topojson geometry 的嵌套 arcs 结构(Polygon: [[ring...]], MultiPolygon: [[[ring...]]]),
    yield 每个 arc 的**绝对索引**。topojson 用 ~a (即 -a-1) 表示 arc a 反向引用 -> 取 a if a>=0 else ~a。"""
    for item in arcs:
        if isinstance(item, list):
            yield from _iter_arc_ids(item)
        else:
            yield item if item >= 0 else ~item


def _giant_adjacent_arcs(topo, g2, giant_vertex_thr, linear_shape_ratio_thr=100000.0):
    """标出"至少一侧邻接**线状网络型**巨型图斑"的 arc(=建筑/道路网那种缠绕细线边界, 平滑只产楔形 sliver)。

    ⚠️ 修正(误伤草地/林地教训): 旧判据=纯顶点数>=阈值, 把草地(maxV 53万)、林地(maxV 5.6万)连成的
    **紧凑大团块**也判 giant -> 绝大多数"田块↔草地/林地"的边被跳过平滑、留成折角, 整体反而不如纯 Chaikin。
    giant-skip 的本意只治**建筑路网**(线状网络: 道路把成百上千田圈成网, 直边对 + Chaikin 在节点产 sliver),
    紧凑大团块(草地/林地)的边界该照常平滑保曲线。

    新判据 = **giant(顶点>=阈值) 且 线状网络**, 二者都满足才跳 Chaikin。线状性用**类无关**的形状比:
      shape_ratio = perimeter^2 / area  (无量纲, 线状细长缠绕 -> 巨大; 紧凑团块 -> 小, ~4π=12.6 起)。
    榆中实测(coverage_simplify tol=5 后, 6 个 giant):
      建筑路网  shape_r=403850 (洞2141)  <- 该跳
      草地团块  shape_r=12284~58613 (洞266~3187) <- 不该跳, 照常 Chaikin
      林地团块  shape_r=2892 (洞335)     <- 不该跳
    形状比把建筑(403850)与最坏草地(58613)拉开 ~7x; 阈值 100000 留 ~1.7x 安全裕度(建筑)/~6.9x(草地)。
    (洞数判不准: 草地 3187 洞 > 建筑 2141 洞, 草地也圈住很多田 -> 不能用洞数, 改用形状比。)
    注: giant 顶点阈值仍是必要前提(单块普通建筑/水体不会有 5 万顶点; 只有路网连通域才 giant), 二者叠加最稳。

    返回: (giant_adj bool[len=n_arcs], n_linear, diag列表)。giant_adj=该 arc 至少一侧是线状巨斑 -> 跳 Chaikin。
    diag: 每个 giant 的 (class_id, verts, holes, shape_ratio, is_linear) 供台账打印。
    topojson 保 object 顺序 = g2 行顺序, 故第 i 个 topo geometry 对应 g2.iloc[i]。
    """
    import shapely
    o = topo.output
    n_arcs = len(o["arcs"])
    geom_objs = o["objects"][list(o["objects"].keys())[0]]["geometries"]
    geoms = g2.geometry.values
    nverts = shapely.get_num_coordinates(geoms)
    cid_col = g2["class_id"].values if "class_id" in g2.columns else None
    giant_adj = np.zeros(n_arcs, dtype=bool)
    n_linear = 0
    diag = []
    for i, gm in enumerate(geom_objs):
        if i >= len(nverts) or nverts[i] < giant_vertex_thr:
            continue
        gobj = geoms[i]
        area = shapely.area(gobj)
        per = shapely.length(gobj)
        shape_ratio = (per * per / area) if area > 0 else float("inf")
        # 洞数(仅诊断, 不入判据)
        gt = getattr(gobj, "geom_type", "")
        if gt == "Polygon":
            holes = len(gobj.interiors)
        elif gt == "MultiPolygon":
            holes = sum(len(p.interiors) for p in gobj.geoms)
        else:
            holes = 0
        is_linear = shape_ratio >= linear_shape_ratio_thr
        diag.append((int(cid_col[i]) if cid_col is not None else -1,
                     int(nverts[i]), int(holes), float(shape_ratio), bool(is_linear)))
        if not is_linear:
            continue            # 紧凑大团块(草地/林地): 不跳, 照常 Chaikin 保曲线
        n_linear += 1
        for aid in _iter_arc_ids(gm.get("arcs", [])):
            if 0 <= aid < n_arcs:
                giant_adj[aid] = True
    return giant_adj, n_linear, diag


def smooth_coverage(gdf, tol=5.0, iters=2, work_crs="EPSG:3857", giant_vertex_thr=50000,
                    linear_shape_ratio_thr=100000.0):
    """全县整体拓扑保持平滑(region-agnostic, 从 yz_smooth2 抽出)+ **per-arc 选择性平滑**。

    gdf(class_id+geometry) ->(work_crs)
      coverage_simplify(tol) 全县整体 去 /N 阶梯, 共享边精确一致, 顶点降 ~2x
      -> topojson.Topology(prequantize=False, shared_coords=False) 全县一次(共享边一份 arc)
      -> **per-arc 选择性 Chaikin**(见下) -> to_gdf 重建 -> 共享边逐点一致
      -> resolve_overlaps 挖掉 Chaikin 残留重叠 -> exact 零重叠无缝。
    返回平滑后 GeoDataFrame(work_crs, 带 class_id)。

    **per-arc 选择性 Chaikin(治"悬空线段"根因)**:
      榆中终版的 sliver 根源 = 建筑类(道路网)被连通域标号成一个巨型多边形(榆中 5.1M 顶点 / 上千洞 =
      被路网圈住的田)。Chaikin 对这种超复杂路网边界逐弧平滑 -> 节点处狂产细 sliver(w<1m), 合并又在
      其边再生 -> 清不动。神池路网稀(洞少)能清, 榆中密(洞多)清不动。
      修法: 标出顶点数 >= giant_vertex_thr **且线状网络**(shape_ratio=perimeter^2/area >= linear_shape_ratio_thr)
      的**线状巨斑**(建筑路网); 对**至少一侧邻接它的 arc 跳过 Chaikin**(道路/建筑本就该直边, 平滑无意义且生楔形),
      只保留 coverage_simplify 的直边; 其余 arc(含邻接草地/林地紧凑大团块的)照常 Chaikin 保曲线。
      ⚠️ 旧版只看顶点数, 把草地(53万顶点)/林地紧凑大团块也判 giant -> 田块↔草地/林地的边被跳平滑留折角,
      反而不如纯 Chaikin。新增形状比判据(类无关)只锁定真·线状路网, 紧凑团块边重新平滑。
      -> 真田块/草地/林地边仍平滑, 仅路网/建筑边界直边, 不再产 sliver。

    为何末步要 resolve_overlaps: topojson.to_gdf 偶尔把极少数内部边(尤其 island/hole 边界)留成
    两条单引用 arc 而非一条共享 arc, Chaikin 后两份独立移动 -> 局部 ~1% 重叠。tol=0 coverage snap
    治不了真重叠(实测无效); STRtree 逐重叠对挖掉(归较大块, union 不变)-> exact 零重叠, 县级可扩展。
    """
    import geopandas as gpd
    from shapely import make_valid, coverage_simplify
    import topojson

    g = gdf.to_crs(work_crs).reset_index(drop=True)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    t0 = time.time()
    simp = coverage_simplify(g.geometry.values, tolerance=tol, simplify_boundary=True)
    simp = np.array([sg if (sg is not None and sg.is_valid) else make_valid(sg) for sg in simp], dtype=object)
    g2 = gpd.GeoDataFrame({"class_id": g["class_id"].values},
                          geometry=gpd.GeoSeries(simp, crs=work_crs))
    g2 = g2[~g2.geometry.is_empty & g2.geometry.notna()].reset_index(drop=True)
    print("[smooth] coverage_simplify(tol=%s) %.0fs | %d polys" % (tol, time.time() - t0, len(g2)), flush=True)

    t1 = time.time()
    topo = topojson.Topology(g2, prequantize=False, shared_coords=False)
    n_arcs = len(topo.output["arcs"])
    print("[smooth] topojson built: %d arcs (%.0fs)" % (n_arcs, time.time() - t1), flush=True)

    if iters > 0:
        # per-arc 选择性平滑: 只对邻接**线状网络型**巨斑(建筑/道路网)的 arc 跳 Chaikin -> 不产楔形 sliver;
        # 紧凑大团块(草地/林地)照常 Chaikin 保曲线(修旧版误伤教训, 见 _giant_adjacent_arcs)。
        giant_adj, n_linear, diag = _giant_adjacent_arcs(
            topo, g2, giant_vertex_thr, linear_shape_ratio_thr=linear_shape_ratio_thr)
        n_skip = int(giant_adj.sum())
        print("[smooth] per-arc selective Chaikin: %d giant parcels (verts>=%d), of which %d 线状(shape_r>=%.0f)->跳; "
              "skip Chaikin on %d/%d arcs (keep straight), smooth %d arcs" %
              (len(diag), giant_vertex_thr, n_linear, linear_shape_ratio_thr,
               n_skip, n_arcs, n_arcs - n_skip), flush=True)
        for (cid_i, v_i, h_i, sr_i, lin_i) in sorted(diag, key=lambda x: -x[3]):
            print("[smooth]   giant: class=%d verts=%d holes=%d shape_ratio=%.1f -> %s" %
                  (cid_i, v_i, h_i, sr_i, "线状(跳Chaikin)" if lin_i else "紧凑(照常平滑)"), flush=True)
        topo.output["arcs"] = [
            (arc if giant_adj[ai] else chaikin_arc(arc, iters))
            for ai, arc in enumerate(topo.output["arcs"])
        ]
    out = topo.to_gdf(crs=work_crs)
    out["geometry"] = out.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    out = out[~out.geometry.is_empty & out.geometry.notna()].reset_index(drop=True)
    # carry class_id through (topojson keeps feature attrs; guard if column name differs)
    if "class_id" not in out.columns and "class_id" in g2.columns and len(out) == len(g2):
        out["class_id"] = g2["class_id"].values

    # 末步: 挖掉 Chaikin 残留重叠 -> exact 无缝。
    if iters > 0:
        t2 = time.time()
        out, n_fixed = resolve_overlaps(out, work_crs=work_crs)
        print("[smooth] resolve_overlaps: %d overlapping pairs fixed (%.0fs)" %
              (n_fixed, time.time() - t2), flush=True)
    print("[smooth] Chaikin(iters=%d, selective) + rebuild + resolve done (%.0fs) | %d polys" %
          (iters, time.time() - t1, len(out)), flush=True)
    return out


# ===========================================================================
# 阶段 3: 可选裁县界(region-agnostic)
# ===========================================================================
def clip_to_boundary(gdf, boundary_geom, utm="EPSG:32648"):
    """真几何 intersection 裁到 boundary_geom(任意闭合县界几何)。
    boundary_geom: shapely 几何(任意 CRS 由 boundary_crs 处理 -> 这里假定调用方已转好或在 utm)。
    gdf 任意 CRS -> utm 做裁切 -> 返回 utm 下裁好的单 Polygon GeoDataFrame。"""
    import geopandas as gpd
    import pandas as pd
    from shapely import make_valid

    g2u = gdf.to_crs(utm).reset_index(drop=True)
    cbgeo = make_valid(boundary_geom)
    minx, miny, maxx, maxy = cbgeo.bounds
    b = g2u.geometry.bounds.values
    keep = ~((b[:, 2] < minx) | (b[:, 0] > maxx) | (b[:, 3] < miny) | (b[:, 1] > maxy))
    g2u = g2u[keep].reset_index(drop=True)
    cov = g2u.geometry.covered_by(cbgeo)
    inside = g2u[cov].copy()
    edge = g2u[~cov].copy()
    rows = []
    for geom, cid in zip(edge.geometry.values, edge["class_id"].values):
        try:
            inter = geom.intersection(cbgeo)
        except Exception:
            try:
                inter = make_valid(geom).intersection(cbgeo)
            except Exception:
                continue
        if inter.is_empty or inter.area <= 0:
            continue
        rows.append({"class_id": cid, "geometry": inter})
    ec = gpd.GeoDataFrame(rows, crs=utm) if rows else gpd.GeoDataFrame(
        {"class_id": [], "geometry": []}, crs=utm)
    out = gpd.GeoDataFrame(pd.concat([inside[["class_id", "geometry"]], ec[["class_id", "geometry"]]],
                                     ignore_index=True), crs=utm)
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    out["geometry"] = out.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    out = out[~out.geometry.is_empty & out.geometry.notna()].reset_index(drop=True)
    return out


# ===========================================================================
# 端到端编排
# ===========================================================================
def default_classes():
    """默认 7 类 schema = dino_parcel_export.CLASSES。"""
    if SIDECAR not in sys.path:
        sys.path.insert(0, SIDECAR)
    from dino_parcel_export import CLASSES
    return [[c[0], c[1], c[2], list(c[3])] for c in CLASSES]


def load_boundary(boundary_arg, utm="EPSG:32648"):
    """boundary 参数 -> (utm 下的) 单个 shapely 几何, 或 None。
    支持 parquet / geojson / 'none'。多 feature -> union。"""
    if not boundary_arg or boundary_arg.lower() == "none":
        return None
    import geopandas as gpd
    from shapely.ops import unary_union
    from shapely import make_valid
    gb = gpd.read_file(boundary_arg) if not boundary_arg.endswith(".parquet") else gpd.read_parquet(boundary_arg)
    gb = gb.to_crs(utm)
    return make_valid(unary_union(gb.geometry.values))


def run_pipeline(mosaic=None, weights=None, backbone=None, out=None, boundary=None, downscale=4,
                 smooth_iters=2, classes=None, gpus=("0", "1", "2", "3"), utm="EPSG:32648",
                 min_dist=20, peak_thr=0.4, min_area_px=200, ridge=True, tol=5.0,
                 save_intermediate=None, device="auto", cells_dir=None, smooth="coverage"):
    """完整通用管线: (mosaic | cells_dir) -> 无缝标准矢量 GeoParquet。返回 (out_gdf, report)。

    device: auto|cuda|mps|cpu。cuda -> 多 GPU 行带并行; mps/cpu -> 单设备整图滑窗。
    cells_dir 给了 -> 先 rasterio.merge 自带拼接成临时 mosaic 再跑。
    """
    classes = classes or default_classes()
    dev = resolve_device(device)
    t0 = time.time()
    # 0) 自带拼接(cells_dir)或用预拼 mosaic
    tmp_mosaic = None
    if cells_dir:
        tmp_mosaic = tempfile.NamedTemporaryFile(suffix="_mosaic.tif", delete=False).name
        mosaic = mosaic_from_cells(cells_dir, tmp_mosaic)
    if not mosaic:
        raise ValueError("run_pipeline 需要 --mosaic 或 --cells-dir 之一")
    try:
        return _run_from_mosaic(mosaic, weights, backbone, out, boundary, downscale, smooth_iters,
                                classes, gpus, utm, min_dist, peak_thr, min_area_px, ridge, tol,
                                save_intermediate, dev, t0, smooth)
    finally:
        if tmp_mosaic and os.path.exists(tmp_mosaic):
            os.remove(tmp_mosaic)


def _run_from_mosaic(mosaic, weights, backbone, out, boundary, downscale, smooth_iters,
                     classes, gpus, utm, min_dist, peak_thr, min_area_px, ridge, tol,
                     save_intermediate, dev, t0, smooth="coverage"):
    # 1) 全局推理 + idmap(cuda 多 GPU; 非 cuda 单设备)
    if str(dev).startswith("cuda") and len(list(gpus)) > 1:
        # band-local + 磁盘 memmap(数值逐位一致, 内存 ~1/带数 -> 大市/省级满卡不爆全图累加器)
        cls, dist, bnd, tr4, crs, cov = infer_global_memmap(mosaic, weights, backbone, downscale, list(gpus))
    else:
        cls, dist, bnd, tr4, crs, cov = infer_single_device(mosaic, weights, backbone, downscale, dev)
    idmap, cls_of = idmap_from_heads(cls, dist, bnd, min_dist, peak_thr, min_area_px, ridge)
    del cls, dist, bnd
    if save_intermediate:
        np.save(str(Path(save_intermediate) / "idmap.npy"), idmap.astype(np.int32))
    # 2) 矢量化 + 平滑(smooth: coverage=全局一次/向后兼容; auto=按规模自动 全局或分块; tiled=强制分块)
    if smooth in ("auto", "tiled"):
        import smooth_dispatch as _sd   # 懒加载:smooth_dispatch 反向 import 本模块,避免循环 import
        if smooth == "tiled":
            smoothed = _sd.tiled_smooth_from_idmap(idmap, cls_of, tr4, crs, tol=tol, iters=smooth_iters)
        else:
            smoothed = _sd.smooth_auto(idmap, cls_of, tr4, crs, tol=tol, iters=smooth_iters)
        del idmap
        print("[pipeline] smoothed(%s): %d polys" % (smooth, len(smoothed)), flush=True)
    else:
        raw = vectorize_idmap(idmap, cls_of, tr4, crs)
        del idmap
        print("[pipeline] vectorized raw: %d polys" % len(raw), flush=True)
        smoothed = smooth_coverage(raw, tol=tol, iters=smooth_iters)
    # 3) 可选裁界
    bnd_geom = load_boundary(boundary, utm) if isinstance(boundary, str) else boundary
    if bnd_geom is not None:
        clipped = clip_to_boundary(smoothed, bnd_geom, utm=utm)
        print("[pipeline] clipped to boundary: %d polys" % len(clipped), flush=True)
    else:
        clipped = smoothed.to_crs(utm)
        print("[pipeline] no boundary -> skip clip", flush=True)
    # 4) 标准后处理
    import postproc
    final, report = postproc.run_postproc(clipped, classes, boundary=bnd_geom, utm=utm)
    final.to_parquet(out)
    report["out"] = out
    report["cov"] = cov
    report["total_min"] = (time.time() - t0) / 60
    from collections import Counter
    cc = Counter(final["label"])
    print("[pipeline] DONE %.0fmin -> %s | %d polys | %s" %
          (report["total_min"], out, len(final), dict(cc)), flush=True)
    return final, report


def main():
    ap = argparse.ArgumentParser(description="region-agnostic 1m parcel vector pipeline")
    ap.add_argument("--mosaic", default="", help="预拼 1m RGB mosaic GeoTIFF(与 --cells-dir 二选一)")
    ap.add_argument("--cells-dir", default="", help="per-cell EPSG:3857 tif 目录(自带 rasterio.merge 拼接)")
    ap.add_argument("--weights", required=True, help="四头 BDDF 权重 .pt")
    ap.add_argument("--backbone", required=True, help="DINOv3-Sat backbone dir")
    ap.add_argument("--out", required=True, help="output GeoParquet")
    ap.add_argument("--boundary", default="none", help="县界 parquet/geojson 或 none")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (auto: cuda>mps>cpu)")
    ap.add_argument("--downscale", type=int, default=4, help="全局累加器 /N 网格(大 mosaic 一遍过)")
    ap.add_argument("--smooth-iters", type=int, default=2, help="Chaikin 迭代次数(0=off)")
    ap.add_argument("--tol", type=float, default=5.0, help="coverage_simplify 容差(米, 3857)")
    ap.add_argument("--classes", default="", help="classes.json (list[[id,zh,en,[r,g,b]]]); 空=默认7类")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--utm", default="EPSG:32648")
    ap.add_argument("--min-dist", type=int, default=20)
    ap.add_argument("--peak-thr", type=float, default=0.4)
    ap.add_argument("--min-area-px", type=int, default=200)
    ap.add_argument("--no-ridge", action="store_true")
    ap.add_argument("--save-intermediate", default="", help="保存 idmap.npy 的目录(可选)")
    ap.add_argument("--smooth", default="coverage", choices=["coverage", "auto", "tiled"],
                    help="平滑路: coverage=全局一次(默认/向后兼容); auto=按地块数自动(>15万转分块,可扩省级); tiled=强制分块")
    a = ap.parse_args()
    if not a.mosaic and not a.cells_dir:
        ap.error("必须给 --mosaic 或 --cells-dir 之一")
    classes = json.loads(Path(a.classes).read_text()) if a.classes else None
    si = a.save_intermediate or None
    if si:
        os.makedirs(si, exist_ok=True)
    run_pipeline(mosaic=a.mosaic or None, cells_dir=a.cells_dir or None,
                 weights=a.weights, backbone=a.backbone, out=a.out, boundary=a.boundary,
                 device=a.device, downscale=a.downscale, smooth_iters=a.smooth_iters, classes=classes,
                 gpus=a.gpus.split(","), utm=a.utm, min_dist=a.min_dist, peak_thr=a.peak_thr,
                 min_area_px=a.min_area_px, ridge=not a.no_ridge, tol=a.tol, save_intermediate=si,
                 smooth=a.smooth)


if __name__ == "__main__":
    main()
