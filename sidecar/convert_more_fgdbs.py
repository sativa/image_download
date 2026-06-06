"""Convert 30 more high-diversity county FGDBs to geoparquet (parallel)."""

import json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd

GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")
OUT_DIR = Path("/tmp/v11_dltb")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def conv(code):
    out = OUT_DIR / f"{code}.parquet"
    if out.exists():
        return code, "cached"
    t = time.time()
    gdb = GDB_ROOT / code / "XYBASE.gdb"
    if not gdb.exists():
        return code, "no gdb"
    g = gpd.read_file(gdb, layer="DLTB", columns=["DLBM"]).to_crs("EPSG:4326")
    g.to_parquet(out, compression="zstd")
    return code, f"{time.time()-t:.1f}s, {len(g)} polys, {out.stat().st_size/1e6:.0f} MB"


if __name__ == "__main__":
    # Existing parquets
    have = {p.stem for p in OUT_DIR.glob("*.parquet")}
    cands = json.loads(Path("/tmp/county_candidates.json").read_text())
    # Counties NOT yet converted, sorted by candidate cell count (more diversity)
    todo = [(c, len(v)) for c, v in cands.items()
            if c not in have and v and "error" not in v[0]]
    todo.sort(key=lambda x: -x[1])
    todo_codes = [c for c, _ in todo[:30]]
    print(f"converting {len(todo_codes)} new counties (have {len(have)} cached)")
    with ProcessPoolExecutor(max_workers=8) as ex:
        for code, info in ex.map(conv, todo_codes):
            print(f"  {code}: {info}")
    final = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"total {len(list(OUT_DIR.glob('*.parquet')))} parquets, {final/1e6:.0f} MB")
