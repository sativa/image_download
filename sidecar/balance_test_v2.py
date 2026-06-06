"""Pick a balanced N-county test set + write new regions JSON.

Each chosen test cell must have all 5 classes with ≥3% area each. Also
ensures test counties are disjoint from train counties.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import box as shp_box

GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")
PQ_DIR = Path("/tmp/v11_dltb")
DLBM_TO_CLASS = {
    "01":1,"02":2,"03":3,"04":4,
    "05":5,"06":5,"07":5,"08":5,"09":5,"10":5,"11":5,"12":5,
}
ID = {1:"耕地", 2:"园地", 3:"林地", 4:"草地", 5:"其他"}
MIN_PCT_PER_CLASS = 3.0
N_TEST = 8


def find_in_county(args):
    code, cells = args
    pq = PQ_DIR / f"{code}.parquet"
    if not pq.exists():
        return code, []
    g = gpd.read_parquet(pq)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try:
        g["geometry"] = g.geometry.make_valid()
    except Exception:
        g["geometry"] = g.geometry.buffer(0)
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
    g = g[g["cid"] > 0]
    if len(g) == 0:
        return code, []
    out = []
    for cell in sorted(cells, key=lambda c: -c.get("score", 0))[:30]:
        bb = tuple(cell["bbox"])
        idx = list(g.sindex.intersection(bb))
        if not idx:
            continue
        try:
            sub = g.iloc[idx].copy()
            sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
            sub = sub[~sub.geometry.is_empty]
            if len(sub) == 0:
                continue
            sub_m = sub.to_crs("EPSG:3857")
            sub_m["area"] = sub_m.geometry.area
            dist = sub_m.groupby(sub["cid"].values)["area"].sum()
        except Exception:
            continue
        total = dist.sum()
        if total <= 0:
            continue
        pct = (dist / total * 100).to_dict()
        if len(pct) < 5:  # need ALL 5 classes
            continue
        if min(pct.values()) < MIN_PCT_PER_CLASS:
            continue
        out.append({"bbox": list(bb), "pct": {ID[k]: v for k, v in pct.items()},
                     "min_pct": min(pct.values())})
    return code, out


def main():
    regions = json.loads(Path("/tmp/v11_regions.json").read_text())
    train_codes = {r["county"] for r in regions["train"]}
    candidates = json.loads(Path("/tmp/county_candidates.json").read_text())
    # Counties not in training, with parquet available
    pq_codes = {p.stem for p in PQ_DIR.glob("*.parquet")}
    eligible = [(c, cells) for c, cells in candidates.items()
                 if c in pq_codes and c not in train_codes and cells
                 and "error" not in (cells[0] if cells else {})]
    print(f"eligible counties to check: {len(eligible)}")
    found = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for code, hits in ex.map(find_in_county, eligible):
            if hits:
                found[code] = hits
    print(f"counties with ≥1 balanced cell: {len(found)}")

    # Pick one most-balanced cell per county; sort by min_pct DESC.
    picks = []
    for code, hits in found.items():
        best = max(hits, key=lambda h: h["min_pct"])
        picks.append((code, best))
    picks.sort(key=lambda x: -x[1]["min_pct"])

    chosen = picks[:N_TEST]
    print(f"\nselected {len(chosen)} balanced test cells:")
    for code, c in chosen:
        s = ", ".join(f"{k}={v:.1f}%" for k, v in c["pct"].items())
        print(f"  {code}: {s}")

    new_test = []
    for k, (code, c) in enumerate(chosen):
        new_test.append({
            "county": code,
            "idx": k,
            "bbox": list(c["bbox"]),
            "gdb": str(GDB_ROOT / code / "XYBASE.gdb"),
        })

    out = {**regions, "test": new_test, "n_test": len(new_test)}
    Path("/tmp/v11_regions_balanced.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote /tmp/v11_regions_balanced.json ({len(new_test)} balanced test cells)")
    print(f"  unique test counties: {sorted({r['county'] for r in new_test})}")


if __name__ == "__main__":
    main()
