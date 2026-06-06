"""Standalone hi-res (Esri/Google z17 ~1m) downloader -> npz. No GDAL (runs on Mac).

Each cell -> {county}_{idx}_{src}.npz with:
  rgb  (H,W,3 uint8) stitched z17 tiles
  ul   [ulx,uly] EPSG:3857 upper-left of the tile mosaic
  res  3857 ground res (m/px)
  bbox [w,s,e,n] WGS84 (the requested cell)
.250 (working rasterio) reprojects/fuses later. Tiles decoded with PIL only.
"""
import argparse, io, json, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

TILE = 256
R = 6378137.0


def tile_xy(lon, lat, z):
    n = 2 ** z
    return (int((lon + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))


def ul_3857(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon * math.pi / 180.0 * R, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R


def one(args):
    bbox, z, src, out = args
    if out.exists():
        return "cached"
    w, s, e, n = bbox
    xmin, ymin = tile_xy(w, n, z)
    xmax, ymax = tile_xy(e, s, z)
    tx, ty = xmax - xmin + 1, ymax - ymin + 1
    cv = np.zeros((ty * TILE, tx * TILE, 3), np.uint8)
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0"

    def fetch(xy):
        x, y = xy
        u = (f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
             if src == "esri" else f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
        try:
            r = sess.get(u, timeout=20)
            r.raise_for_status()
            return x, y, r.content
        except Exception:
            return x, y, None

    ok = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for x, y, d in ex.map(fetch, [(x, y) for y in range(ymin, ymax + 1) for x in range(xmin, xmax + 1)]):
            if d:
                try:
                    cv[(y - ymin) * TILE:(y - ymin + 1) * TILE,
                       (x - xmin) * TILE:(x - xmin + 1) * TILE] = np.array(Image.open(io.BytesIO(d)).convert("RGB"))
                    ok += 1
                except Exception:
                    pass
    if ok == 0:
        return "FAIL"
    ulx, uly = ul_3857(xmin, ymin, z)
    res = 2 * math.pi * R / (2 ** z) / TILE
    np.savez_compressed(out, rgb=cv, ul=np.array([ulx, uly]), res=res, bbox=np.array(bbox))
    return f"ok{ok}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--source", default="esri")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cells = json.loads(a.cells_json.read_text())
    jobs = [(tuple(c["bbox"]), a.zoom, a.source,
             a.out_dir / f'{c["county"]}_{c["idx"]}_{a.source}.npz') for c in cells]
    jobs = [j for j in jobs if not j[3].exists()]
    print(f"download {len(jobs)} cells z{a.zoom} {a.source} -> {a.out_dir}", flush=True)
    t0 = time.time(); done = 0; fails = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for fu in as_completed(futs):
            st = fu.result(); done += 1; fails += (st == "FAIL")
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} fails={fails} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] {done} cells, {fails} fails, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
