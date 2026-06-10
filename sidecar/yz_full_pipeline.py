"""Yuzhong FULL-COUNTY end-to-end pipeline (stability test of the whole chain):
CLI-downloaded cells -> 6x6-cell block MOSAICS (rasterio.merge, single-image watershed per block ->
no within-block seams) -> parcel_dist backend per block (four-head bddf ckpt: cls+bnd+dist+FFL
regularized vectors) -> concat one county GeoParquet + per-class stats.

Run on .250. Blocks processed sequentially on one GPU; each failure is logged and skipped (stability:
one bad block must not kill the county run)."""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="/tmp/yz_full_regions.json")
    ap.add_argument("--tif-dir", default="/mnt/sda/zf/landform/data/yz_full_tif")
    ap.add_argument("--work", default="/mnt/sda/zf/landform/results/yz_full")
    ap.add_argument("--weights", default="/mnt/sda/zf/landform/results/dino_v3_bddf/last.pt")
    ap.add_argument("--backbone", default="/home/ps/landform/dinov3/dinov3-vitl16-sat493m")
    ap.add_argument("--block", type=int, default=6, help="block = NxN cells mosaicked into one image")
    ap.add_argument("--python", default="/home/ps/miniconda3/bin/python")
    ap.add_argument("--out", default="/mnt/sda/zf/landform/results/yuzhong_full_region.parquet")
    a = ap.parse_args()
    import rasterio
    from rasterio.merge import merge
    t0 = time.time()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    cells = json.loads(Path(a.regions).read_text())
    td = Path(a.tif_dir)
    # group cells into blocks by grid position
    blocks = {}
    for c in cells:
        f = td / f"{c['county']}_{c['idx']}_esri.tif"
        if not f.exists():
            continue
        blocks.setdefault((c["col"] // a.block, c["row"] // a.block), []).append(f)
    print(f"[yzfull] {sum(len(v) for v in blocks.values())} cells -> {len(blocks)} blocks", flush=True)
    ok = failed = 0
    for bi, (key, files) in enumerate(sorted(blocks.items())):
        tag = f"b{key[0]:02d}_{key[1]:02d}"
        bp = work / f"{tag}.parquet"
        if bp.exists():
            ok += 1; continue                                     # resume support
        try:
            mt = work / f"{tag}_mosaic.tif"
            if not mt.exists():                                   # 1) mosaic (拼接)
                srcs = [rasterio.open(str(f)) for f in files]
                mosaic, tr = merge(srcs)
                meta = srcs[0].meta.copy()
                meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=tr, compress="jpeg")
                with rasterio.open(mt, "w", **meta) as d:
                    d.write(mosaic)
                for s in srcs:
                    s.close()
            # 2) final model -> FFL polygons (parcel_dist backend, four-head ckpt)
            r = subprocess.run([a.python, "-m", "sam3_classify", "--backend", "parcel_dist",
                                "--input", str(mt), "--output", str(work / f"{tag}.tif"),
                                "--weights", a.weights, "--backbone-dir", a.backbone, "--device", "cuda"],
                               cwd="/home/ps/landform/sidecar", capture_output=True, text=True, timeout=3600)
            if not (work / f"{tag}.parquet").exists():
                raise RuntimeError(f"no parquet; tail={r.stdout[-300:]}")
            mt.unlink(missing_ok=True)                            # free disk (mosaic ~200MB/block)
            ok += 1
            print(f"  [{bi+1}/{len(blocks)}] {tag}: {len(files)} cells OK ({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            failed += 1
            print(f"  [{bi+1}/{len(blocks)}] {tag}: FAILED {str(ex)[:200]}", flush=True)
    # 3) concat county GeoParquet
    import geopandas as gpd, pandas as pd
    gdfs = []
    for p in sorted(work.glob("b*.parquet")):
        try:
            d = gpd.read_parquet(p); d["block"] = p.stem; gdfs.append(d)
        except Exception as ex:
            print(f"  concat skip {p.name}: {ex}", flush=True)
    reg = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    reg.insert(0, "gid", range(1, len(reg) + 1))
    reg.to_parquet(a.out)
    from collections import Counter
    cc = Counter(reg["label"]); ar = reg.groupby("label")["area_m2"].sum().div(1e6).round(1)
    print(f"[yzfull] DONE blocks ok={ok} failed={failed} | {len(reg)} parcels -> {a.out}", flush=True)
    print(f"  counts: {dict(cc)}", flush=True)
    print(f"  km2: {ar.to_dict()}", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
