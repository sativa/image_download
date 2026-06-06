"""Rank counties by cropland share and pick the top N for v17 training.

For each county parquet:
  1. Sum total polygon area
  2. Sum cropland (DLBM 01XX) + orchard (02XX) area
  3. Cropland_share = (耕地+园地) / total

Output:
  - Top 40 cropland-rich counties (share > 0.50)
  - For each, 3 best cells (cropland-rich + diverse)
  - Held-out test: still the 8 balanced cells (we want apples-to-apples eval)
"""

from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
from shapely.geometry import box as shp_box

PQ_DIR = Path("/tmp/v11_dltb")
CAND = Path("/tmp/county_candidates.json")
BAL = Path("/tmp/v11_regions_balanced.json")
OUT = Path("/tmp/v17_regions.json")
GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")

DLBM = {"01":1,"02":2,"03":3,"04":4,"05":5,"06":5,"07":5,"08":5,"09":5,"10":5,"11":5,"12":5}
N_COUNTIES = 35
N_PER_COUNTY = 3


def county_stats(code: str):
    """Return (code, cropland_share, total_area_m2, n_polys, top_cells)."""
    pq = PQ_DIR / f"{code}.parquet"
    if not pq.exists():
        return code, 0.0, 0.0, 0, []
    g = gpd.read_parquet(pq)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    try: g["geometry"] = g.geometry.make_valid()
    except Exception: g["geometry"] = g.geometry.buffer(0)
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM).fillna(0).astype(int)
    g_m = g.to_crs("EPSG:3857").copy()
    g_m["area"] = g_m.geometry.area
    total = float(g_m["area"].sum())
    crop = float(g_m.loc[g_m["cid"].isin([1, 2]), "area"].sum())
    share = crop / total if total > 0 else 0
    return code, share, total, len(g), g  # return g for cell selection later


def best_cells_for(code: str, g, candidates: list) -> list:
    """Pick top N cells in this county with high cropland density."""
    g_m = g.to_crs("EPSG:3857")
    g_m["cid"] = g["cid"]
    g_m["area"] = g_m.geometry.area
    scored = []
    for cell in sorted(candidates, key=lambda c: -c.get("score", 0))[:30]:
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
            tot = sub_m["area"].sum()
            crop = sub_m.loc[sub["cid"].isin([1, 2]), "area"].sum()
            if tot <= 0:
                continue
            scored.append({"bbox": list(bb), "cropland_share": crop / tot,
                            "n_classes": sub["cid"].nunique()})
        except Exception:
            continue
    # Want cropland-rich AND multi-class (to give context for boundary)
    scored.sort(key=lambda c: (-c["cropland_share"] - 0.1 * c["n_classes"]))
    return scored[:N_PER_COUNTY]


def main():
    cands = json.loads(CAND.read_text())
    balanced = json.loads(BAL.read_text())
    test_counties = {r["county"] for r in balanced["test"]}
    have_pq = {p.stem for p in PQ_DIR.glob("*.parquet")}
    # Skip test counties & those without parquet
    eligible = sorted(have_pq - test_counties)
    print(f"computing cropland share for {len(eligible)} counties ...")

    stats = []
    for code in eligible:
        try:
            code_, share, total, n_polys, g = county_stats(code)
            stats.append((code_, share, total, n_polys, g))
        except Exception as e:
            print(f"  {code} failed: {e}")
    stats.sort(key=lambda x: -x[1])
    print("\ntop 50 cropland-rich counties:")
    for code, share, total, n_polys, _ in stats[:50]:
        print(f"  {code}: cropland_share={share:.3f}, polys={n_polys:,}")

    # Pick top 35 (with cropland share > 0.30 sanity check)
    chosen = [s for s in stats if s[1] > 0.30][:N_COUNTIES]
    print(f"\nselected {len(chosen)} counties (share > 0.30)")

    train_regions = []
    for code, share, _, _, g in chosen:
        cells = cands.get(code, [])
        if not cells:
            continue
        best_cells = best_cells_for(code, g, cells)
        for k, c in enumerate(best_cells):
            train_regions.append({
                "county": code,
                "idx": k,
                "bbox": c["bbox"],
                "gdb": str(GDB_ROOT / code / "XYBASE.gdb"),
                "cropland_share_cell": c["cropland_share"],
                "county_share": share,
            })

    out = {
        "train": train_regions,
        "test": balanced["test"],
        "n_train": len(train_regions),
        "n_test": len(balanced["test"]),
        "n_train_counties": len(chosen),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT} ({len(train_regions)} train regions across {len(chosen)} counties)")


if __name__ == "__main__":
    main()
