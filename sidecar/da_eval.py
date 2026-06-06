"""Evaluate Delineate-Anything (YOLOv11-seg field-boundary foundation model, FBIS-22M) on Tibet tiles,
vs FSDA reference parcels — same boundary-F1 metric as boundary_eval.py, to decide whether to adopt
DA as the delineation layer in a multi-model coupled backend (DA boundaries + our DINOv3 classifier)."""
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np
import rasterio
import cv2
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box as shp_box
from ultralytics import YOLO

import os
DA = Path(os.environ.get("DA_CKPT", "/Users/zhangfeng/D/delineate_anything/DelineateAnything.pt"))
TIF = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/z17_tibet")
SHP = "/Users/zhangfeng/Downloads/xizang_fsda/Xizang cropland datasets/ALL_Xizang_Albers.shp"
REG = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/tibet_regions.json"
WIN = 1024; N_CELLS = 12
import os
DOWN = int(os.environ.get("DA_DOWN", "1"))   # downsample factor before DA (1=native 1m, 3=~3m sweet spot)


def da_instances(model, rgb):
    """Tile the image, run DA per window, stitch into a global instance-id map."""
    H, W, _ = rgb.shape
    idmap = np.zeros((H, W), np.int32); nid = 0
    step = WIN * DOWN
    for y in range(0, H, step):
        for x in range(0, W, step):
            win = rgb[y:y + step, x:x + step]
            wh, ww = win.shape[:2]
            inp = cv2.resize(win, (ww // DOWN, wh // DOWN), interpolation=cv2.INTER_AREA) if DOWN > 1 else win
            r = model.predict(inp, imgsz=max(320, (max(inp.shape[:2]) // 32) * 32), conf=0.25,
                              verbose=False, retina_masks=True)[0]
            if r.masks is None:
                continue
            for m in r.masks.data.cpu().numpy():
                mb = cv2.resize(m, (ww, wh), interpolation=cv2.INTER_NEAREST) > 0.5
                nid += 1; sub = idmap[y:y + wh, x:x + ww]
                sub[mb & (sub == 0)] = nid
    return idmap


def edges(idmap):
    b = np.zeros(idmap.shape, bool)
    d = idmap[:-1, :] != idmap[1:, :]; b[:-1, :] |= d; b[1:, :] |= d
    d = idmap[:, :-1] != idmap[:, 1:]; b[:, :-1] |= d; b[:, 1:] |= d
    return b.astype(np.uint8)


def bf1(pred_b, true_b, tol=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    tdil = cv2.dilate(true_b, k) > 0; pdil = cv2.dilate(pred_b, k) > 0
    pb = pred_b > 0; tb = true_b > 0
    prec = (pb & tdil).sum() / max(1, pb.sum()); rec = (tb & pdil).sum() / max(1, tb.sum())
    return 2 * prec * rec / max(1e-9, prec + rec), prec, rec


def main():
    model = YOLO(str(DA)); print("DA loaded", flush=True)
    print("loading FSDA -> 3857 ...", flush=True)
    fsda = gpd.read_file(SHP).to_crs("EPSG:3857"); sidx = fsda.sindex
    regions = json.load(open(REG))
    names = [f"XZ_{r['idx']}" for r in regions][-N_CELLS:]   # use the held-out tail
    sf1 = sp = sr = 0.0; nparc = 0; n = 0
    for nm in names:
        ep = TIF / f"{nm}_esri.tif"
        if not ep.exists(): continue
        with rasterio.open(ep) as s:
            rgb = np.transpose(s.read()[:3], (1, 2, 0)); H, W = s.height, s.width; tr = s.transform; bn = s.bounds
        idm = da_instances(model, np.ascontiguousarray(rgb))
        nparc += int(idm.max())
        pb = edges(idm)
        # FSDA reference parcel edges
        idx = list(sidx.intersection((bn.left, bn.bottom, bn.right, bn.top)))
        ref_id = np.zeros((H, W), np.int32)
        if idx:
            sub = fsda.iloc[idx].reset_index(drop=True); cb = shp_box(bn.left, bn.bottom, bn.right, bn.top)
            shp = [(g, j + 1) for j, g in enumerate(sub.geometry) if g.intersects(cb)]
            if shp: ref_id = rasterize(shp, out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        tb = edges(ref_id)
        f1, pr, rc = bf1(pb, tb, 3); sf1 += f1; sp += pr; sr += rc; n += 1
        print(f"  {nm}: DA parcels={int(idm.max())} boundary-F1(tol3)={f1:.3f}", flush=True)
    print(f"\n=== Delineate-Anything 边界质量 (Tibet {n} cells, vs FSDA, tol=3px) ===", flush=True)
    print(f"  boundary-F1={sf1/n:.4f}  P={sp/n:.3f}  R={sr/n:.3f}  avg parcels/cell={nparc/n:.0f}", flush=True)
    print(f"  (对比: 我们边界头 Gansu/DLTB tol3 = 0.549)", flush=True)


if __name__ == "__main__":
    main()
