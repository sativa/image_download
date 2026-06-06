"""Convert Tibet tiles + FSDA parcels into a YOLOv11-seg dataset to FINE-TUNE Delineate-Anything
(strengthen it on our parcels -> a coupled delineation module). Each 1024-px window -> a PNG image +
a label .txt (one line per parcel: class + normalized polygon vertices). Single class 'field'.
Split by cell: 90 train / 30 val (matches c_1m_tibet)."""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import rasterio
import cv2
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box as shp_box

TIF = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/z17_tibet")
SHP = "/Users/zhangfeng/Downloads/xizang_fsda/Xizang cropland datasets/ALL_Xizang_Albers.shp"
REG = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/tibet_regions.json"
ROOT = Path("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/yolo_tibet")
WIN = 1024; N_VAL = 30
for sp in ("train", "val"):
    (ROOT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (ROOT / "labels" / sp).mkdir(parents=True, exist_ok=True)

print("loading FSDA -> 3857 ...", flush=True)
fsda = gpd.read_file(SHP).to_crs("EPSG:3857"); sidx = fsda.sindex
regions = json.load(open(REG)); names = [f"XZ_{r['idx']}" for r in regions]
val_names = set(names[-N_VAL:]); nimg = 0
for nm in names:
    ep = TIF / f"{nm}_esri.tif"
    if not ep.exists(): continue
    sp = "val" if nm in val_names else "train"
    with rasterio.open(ep) as s:
        rgb = np.transpose(s.read()[:3], (1, 2, 0)); H, W = s.height, s.width; tr = s.transform; bn = s.bounds
    idx = list(sidx.intersection((bn.left, bn.bottom, bn.right, bn.top)))
    if not idx: continue
    sub = fsda.iloc[idx].reset_index(drop=True); cb = shp_box(bn.left, bn.bottom, bn.right, bn.top)
    polys = [g for g in sub.geometry if g.intersects(cb)]
    inv = ~tr                                                   # world->pixel
    for y in range(0, H, WIN):
        for x in range(0, W, WIN):
            win = rgb[y:y + WIN, x:x + WIN]; wh, ww = win.shape[:2]
            if wh < 64 or ww < 64: continue
            wb = shp_box(*(tr * (x, y + wh)), *(tr * (x + ww, y)))   # window bounds in world
            lines = []
            for g in polys:
                gi = g.intersection(wb)
                if gi.is_empty or gi.area < 20: continue
                geoms = list(gi.geoms) if gi.geom_type.startswith("Multi") else [gi]
                for gg in geoms:
                    if gg.geom_type != "Polygon" or gg.area < 20: continue
                    xs, ys = gg.exterior.coords.xy
                    px = [(inv * (cx, cy)) for cx, cy in zip(xs, ys)]   # world->pixel
                    norm = []
                    for cxp, cyp in px:
                        u = (cxp - x) / ww; v = (cyp - y) / wh
                        norm += [f"{min(1,max(0,u)):.5f}", f"{min(1,max(0,v)):.5f}"]
                    if len(norm) >= 6: lines.append("0 " + " ".join(norm))
            if not lines: continue
            base = f"{nm}_{y}_{x}"
            cv2.imwrite(str(ROOT / "images" / sp / f"{base}.png"), cv2.cvtColor(win, cv2.COLOR_RGB2BGR))
            (ROOT / "labels" / sp / f"{base}.txt").write_text("\n".join(lines))
            nimg += 1
    if nimg and nimg % 50 == 0: print(f"  {nimg} windows", flush=True)

(ROOT / "data.yaml").write_text(
    f"path: {ROOT}\ntrain: images/train\nval: images/val\nnames:\n  0: field\n")
print(f"done: {nimg} windows -> {ROOT}; data.yaml written", flush=True)
