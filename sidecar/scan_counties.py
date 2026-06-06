"""Scan the 104 Gansu counties' DLTB FGDBs to find diverse 2-km training cells.

For each county we sample candidate 0.02° × 0.02° cells, score them by:
  * number of distinct 一级地类 classes present (we want ≥4)
  * presence of less-common classes (water, built-up) gets a bonus

Result: a JSON list of bboxes + county code, ready for download.

Reading 104 FGDBs serially would take ~30 min; we parallelise over
counties via ProcessPool (fiona/pyogrio are GIL-bound in C calls but
ProcessPool sidesteps that).
"""

from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")
DLBM_TO_CLASS = {  # GB/T 21010 first 2 digits → 一级地类
    "01": 1, "02": 2, "03": 3, "04": 4,
    "05": 5, "06": 5, "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5,
}


def scan_county(county_dir: Path):
    """Return (county_code, candidate_bboxes_with_score)."""
    code = county_dir.name
    gdb_path = county_dir / "XYBASE.gdb"
    if not gdb_path.exists():
        return code, []
    try:
        import geopandas as gpd
        g = gpd.read_file(gdb_path, layer="DLTB", columns=["DLBM"])
    except Exception as e:
        return code, [{"error": str(e)[:200]}]
    if not len(g):
        return code, []
    g = g.to_crs("EPSG:4326")
    g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
    g = g[g["cid"] > 0]
    if not len(g):
        return code, []
    W, S, E, N = g.total_bounds
    # Grid the county and score each 0.02° cell.
    step = 0.02
    sindex = g.sindex
    candidates = []
    for ny in np.arange(S, N - step, step):
        for nx in np.arange(W, E - step, step):
            bb = (nx, ny, nx + step, ny + step)
            idx = list(sindex.intersection(bb))
            if not idx:
                continue
            sub_cids = g.iloc[idx]["cid"]
            uniq = sub_cids.unique()
            if len(uniq) < 4:
                continue
            # Score: class count + bonus for rare classes 5
            score = len(uniq) + (1 if 5 in uniq else 0)
            candidates.append({"bbox": [float(x) for x in bb],
                                "n_classes": int(len(uniq)),
                                "n_polys": int(len(idx)),
                                "score": int(score)})
    return code, candidates


def main():
    counties = sorted(p for p in GDB_ROOT.iterdir()
                      if p.is_dir() and p.name.isdigit())
    print(f"scanning {len(counties)} counties in parallel ...")
    results = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(scan_county, c): c.name for c in counties}
        for fut in as_completed(futs):
            code, cands = fut.result()
            results[code] = cands
            print(f"  {code}: {len(cands)} candidate cells")

    out = Path("/tmp/county_candidates.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")
    # Summary.
    n_ok = sum(1 for v in results.values() if v and "error" not in v[0])
    n_total_cells = sum(len(v) for v in results.values() if "error" not in (v[0] if v else {}))
    print(f"  counties scanned ok: {n_ok}/{len(results)}")
    print(f"  total candidate cells: {n_total_cells:,}")


if __name__ == "__main__":
    main()
