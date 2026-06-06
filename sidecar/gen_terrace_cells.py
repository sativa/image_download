"""Generate TERRACE-rich new training cells from the full Gansu DLTB (runs on .174).

Targets the cropland-recognition weak spot: loess-plateau terraces (梯田, GDLX='TT') and slope
cropland (坡地, GDLX='PD'). Bins terrace/slope parcels into a 0.02 deg grid (~c_1m cell size),
scores each grid cell by terrace-parcel density, excludes grids already in c_1m, and emits the
top-N as a region JSON ({county, idx, bbox}) for the 1m tile downloader.
"""
import argparse, json, math
from collections import Counter
from pathlib import Path

import numpy as np
import geopandas as gpd

DLTB = "/mnt/sdb/shared/zf/gs_landuse/gs_landuse_DLTB.parquet"
GRID = 0.02


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c1m-centroids", default="/tmp/c1m_centroids.json")
    p.add_argument("--n", type=int, default=2000, help="number of new terrace cells to emit")
    p.add_argument("--out", default="/tmp/terrace_cells.json")
    a = p.parse_args()

    print("[gen] reading 梯田(TT)+坡地(PD) parcels ...", flush=True)
    g = gpd.read_parquet(DLTB, columns=["GDLX", "QSDWDM", "geometry"],
                         filters=[("GDLX", "in", ["TT", "PD"])])
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    print(f"[gen] {len(g)} terrace/slope parcels", flush=True)
    c = g.geometry.centroid
    gx = np.floor(c.x.values / GRID).astype(int)
    gy = np.floor(c.y.values / GRID).astype(int)
    is_tt = (g["GDLX"].values == "TT").astype(int)
    dw = g["QSDWDM"].astype(str).str[:6].values

    # exclude grids already covered by c_1m
    ex = set()
    for lon, lat in json.load(open(a.c1m_centroids)):
        ex.add((int(math.floor(lon / GRID)), int(math.floor(lat / GRID))))

    cells = {}  # (gx,gy) -> [n_total, n_tt, Counter(county)]
    for i in range(len(gx)):
        k = (int(gx[i]), int(gy[i]))
        if k in ex:
            continue
        e = cells.get(k)
        if e is None:
            cells[k] = e = [0, 0, Counter()]
        e[0] += 1; e[1] += int(is_tt[i]); e[2][dw[i]] += 1

    # rank by terrace (TT) count, then total; require a meaningful density
    ranked = sorted(cells.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True)
    out = []
    for (cx, cy), (ntot, ntt, ccnt) in ranked:
        if ntt < 20:                      # skip sparse cells (need real terrace density)
            continue
        county = ccnt.most_common(1)[0][0]
        out.append({"county": county, "idx": int(cx * 100000 + (cy % 100000)),
                    "bbox": [round(cx * GRID, 6), round(cy * GRID, 6),
                             round((cx + 1) * GRID, 6), round((cy + 1) * GRID, 6)],
                    "n_terrace": int(ntt), "n_slope_total": int(ntot)})
        if len(out) >= a.n:
            break
    json.dump(out, open(a.out, "w"))
    tt_total = sum(o["n_terrace"] for o in out)
    print(f"[gen] emitted {len(out)} terrace cells -> {a.out} | total 梯田 parcels covered {tt_total}", flush=True)
    print(f"[gen] sample: {out[0] if out else None}", flush=True)
    cc = Counter(o["county"] for o in out)
    print(f"[gen] top counties: {cc.most_common(8)}", flush=True)


if __name__ == "__main__":
    main()
