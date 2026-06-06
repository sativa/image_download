"""Generate 园地(orchard, DLBM 02)-dense new training cells from the full Gansu DLTB (runs on .174).
Bins orchard parcels into a 0.02 deg grid, excludes cells already used, emits top-N (>=min) as a
region JSON {county, idx, bbox, n_orchard} for the 1m tile downloader."""
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import geopandas as gpd

DLTB = "/mnt/sdb/shared/zf/gs_landuse/gs_landuse_DLTB.parquet"
GRID = 0.02


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--used", default="/tmp/used_centroids.json")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--min", type=int, default=50)
    p.add_argument("--out", default="/tmp/orchard_cells.json")
    a = p.parse_args()

    g = gpd.read_parquet(DLTB, columns=["DLBM", "QSDWDM", "geometry"])
    dlbm = g["DLBM"].astype(str).str.zfill(4)
    g = g[dlbm.str[:2] == "02"].reset_index(drop=True)
    c = g.geometry.centroid
    gx = np.floor(c.x.values / GRID).astype(int)
    gy = np.floor(c.y.values / GRID).astype(int)
    dw = g["QSDWDM"].astype(str).str[:6].values

    ex = set()
    for lon, lat in json.load(open(a.used)):
        ex.add((int(math.floor(lon / GRID)), int(math.floor(lat / GRID))))

    cells = {}
    for i in range(len(gx)):
        k = (int(gx[i]), int(gy[i]))
        if k in ex:
            continue
        e = cells.get(k)
        if e is None:
            cells[k] = e = [0, Counter()]
        e[0] += 1
        e[1][dw[i]] += 1

    ranked = sorted(cells.items(), key=lambda kv: kv[1][0], reverse=True)
    out = []
    for (cx, cy), (n, cc) in ranked:
        if n < a.min:
            break
        out.append({"county": cc.most_common(1)[0][0],
                    "idx": int(cx * 100000 + (cy % 100000)),
                    "bbox": [round(cx * GRID, 6), round(cy * GRID, 6),
                             round((cx + 1) * GRID, 6), round((cy + 1) * GRID, 6)],
                    "n_orchard": int(n)})
        if len(out) >= a.n:
            break
    json.dump(out, open(a.out, "w"))
    ntot = sum(o["n_orchard"] for o in out)
    print("emitted %d orchard cells (>=%d), orchard parcels covered %d" % (len(out), a.min, ntot))
    print("sample:", out[0] if out else None)
    cc = Counter(o["county"] for o in out)
    print("top counties:", cc.most_common(6))


if __name__ == "__main__":
    main()
