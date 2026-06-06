"""Mine cells dense in the RARE 7-class land-cover classes from the full Gansu DLTB (runs on .174),
to rebalance the 7-class training set. Targets:
   园地 DLBM 02  (7-class id 2)
   水体 DLBM 11  (id 5)
   荒漠 DLBM 12  (id 7, 其他土地)
For each target: bin parcels into a 0.02 deg grid, exclude already-used cells, rank by that class's
parcel count, take the top n-per (>= min). Combine across classes, dedup by cell. Emits one region
JSON {county, idx, bbox, rare_class, n_rare} for the 1m tile downloader."""
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import geopandas as gpd

DLTB = "/mnt/sdb/shared/zf/gs_landuse/gs_landuse_DLTB.parquet"
GRID = 0.02
TARGETS = [("02", "园地"), ("11", "水体"), ("12", "荒漠")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--used", default="/tmp/used_centroids.json")
    p.add_argument("--n-per", type=int, default=1300, help="max cells per rare class")
    p.add_argument("--min", type=int, default=20, help="min parcels of that class in a cell")
    p.add_argument("--out", default="/tmp/rare_cells.json")
    a = p.parse_args()

    g = gpd.read_parquet(DLTB, columns=["DLBM", "QSDWDM", "geometry"])
    dlbm = g["DLBM"].astype(str).str.zfill(4)
    pre = dlbm.str[:2].values
    c = g.geometry.centroid
    gx = np.floor(c.x.values / GRID).astype(int)
    gy = np.floor(c.y.values / GRID).astype(int)
    dw = g["QSDWDM"].astype(str).str[:6].values

    ex = set()
    for lon, lat in json.load(open(a.used)):
        ex.add((int(math.floor(lon / GRID)), int(math.floor(lat / GRID))))

    out = []
    chosen = set()
    for code, name in TARGETS:
        m = pre == code
        cells = {}
        gxm, gym, dwm = gx[m], gy[m], dw[m]
        for i in range(len(gxm)):
            k = (int(gxm[i]), int(gym[i]))
            if k in ex or k in chosen:
                continue
            e = cells.get(k)
            if e is None:
                cells[k] = e = [0, Counter()]
            e[0] += 1
            e[1][dwm[i]] += 1
        ranked = sorted(cells.items(), key=lambda kv: kv[1][0], reverse=True)
        added = 0
        for (cx, cy), (n, cc) in ranked:
            if n < a.min or added >= a.n_per:
                break
            chosen.add((cx, cy))
            out.append({"county": cc.most_common(1)[0][0],
                        "idx": int(cx * 100000 + (cy % 100000)),
                        "bbox": [round(cx * GRID, 6), round(cy * GRID, 6),
                                 round((cx + 1) * GRID, 6), round((cy + 1) * GRID, 6)],
                        "rare_class": name, "n_rare": int(n)})
            added += 1
        print("%s (DLBM %s): %d cells (>=%d)" % (name, code, added, a.min))

    json.dump(out, open(a.out, "w"))
    print("TOTAL %d cells -> %s" % (len(out), a.out))
    print("by class:", Counter(o["rare_class"] for o in out))


if __name__ == "__main__":
    main()
