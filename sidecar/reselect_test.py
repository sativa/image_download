"""Pick a BALANCED test set: 4 cells across 4 counties, each cell containing
≥4 of 5 classes with ≥3% area each, AND not in any training county.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box as shp_box

GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")
DLBM_TO_CLASS = {
    "01": 1, "02": 2, "03": 3, "04": 4,
    "05": 5, "06": 5, "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5,
}
ID_TO_DLTB = {1:"耕地", 2:"园地", 3:"林地", 4:"草地", 5:"其他"}

regions = json.loads(Path("/tmp/v11_regions.json").read_text())
candidates = json.loads(Path("/tmp/county_candidates.json").read_text())
train_counties = {r["county"] for r in regions["train"]}

balanced = []  # (county, bbox, class_pct_dict)

# Iterate all non-training counties, find a cell with balanced classes.
non_train = [c for c in candidates if c not in train_counties and candidates[c]]
print(f"checking {len(non_train)} non-training counties for balanced cells")

for code in non_train:
    cells = candidates[code]
    if not cells or "error" in (cells[0] if cells else {}):
        continue
    # Load this county once.
    pq = Path(f"/tmp/v11_dltb/{code}.parquet")
    if not pq.exists():
        continue  # only check counties we already have parquets for
    g = gpd.read_parquet(pq)
    g = g.to_crs("EPSG:3857")  # metric area
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
    g = g[g["cid"] > 0]
    if len(g) == 0:
        continue
    # For each candidate cell, compute class distribution
    g_wgs = g.to_crs("EPSG:4326")
    for c in sorted(cells, key=lambda x: -x.get("score", 0))[:20]:
        bb = tuple(c["bbox"])
        idx = list(g_wgs.sindex.intersection(bb))
        if not idx:
            continue
        sub = g_wgs.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bb))
        sub = sub[~sub.geometry.is_empty].to_crs("EPSG:3857")
        sub["area"] = sub.geometry.area
        dist = sub.groupby("cid")["area"].sum()
        total = dist.sum()
        if total <= 0:
            continue
        pct = (dist / total * 100)
        # Need ≥4 classes with ≥3% AND all 5 classes present
        ge3 = (pct >= 3).sum()
        n_classes = len(pct)
        # Score: prefer cells with all 5 classes balanced
        min_pct = pct.min()
        if ge3 >= 4 and n_classes == 5:
            balanced.append({
                "county": code,
                "bbox": bb,
                "n_classes": int(n_classes),
                "ge3_classes": int(ge3),
                "min_pct": float(min_pct),
                "pct": {ID_TO_DLTB[c]: float(p) for c, p in pct.items()},
            })

print(f"\nfound {len(balanced)} balanced test candidates")
# Pick 4 counties, top 1 cell each (sort by min_pct = most balanced)
balanced.sort(key=lambda x: -x["min_pct"])
seen = set()
chosen = []
for cell in balanced:
    if cell["county"] in seen:
        continue
    seen.add(cell["county"])
    chosen.append(cell)
    if len(chosen) == 4:
        break

print(f"\nselected 4 balanced test cells:")
for c in chosen:
    pct_str = ", ".join(f"{k}={v:.1f}%" for k, v in c["pct"].items())
    print(f"  {c['county']}: {pct_str}")

# Build new regions JSON: same train, replaced test.
new_test = []
for k, c in enumerate(chosen):
    new_test.append({
        "county": c["county"],
        "idx": k,
        "bbox": list(c["bbox"]),
        "gdb": str(GDB_ROOT / c["county"] / "XYBASE.gdb"),
    })

out = {**regions, "test": new_test, "n_test": len(new_test)}
Path("/tmp/v11_regions_balanced.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\nwrote /tmp/v11_regions_balanced.json")
