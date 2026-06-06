"""Convert the 24 needed county FGDBs to compact geoparquet."""

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
        return code, out, "cached"
    t = time.time()
    gdb = GDB_ROOT / code / "XYBASE.gdb"
    g = gpd.read_file(gdb, layer="DLTB", columns=["DLBM"]).to_crs("EPSG:4326")
    g.to_parquet(out, compression="zstd")
    return code, out, f"{time.time()-t:.1f}s, {len(g)} polys, {out.stat().st_size/1e6:.0f} MB"


if __name__ == "__main__":
    regions = json.loads(Path("/tmp/v11_regions.json").read_text())
    codes = sorted({r["county"] for r in regions["train"] + regions["test"]})
    print(f"converting {len(codes)} FGDBs in parallel ...")
    with ProcessPoolExecutor(max_workers=8) as ex:
        for code, out, info in ex.map(conv, codes):
            print(f"  {code}: {info}")
    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"total {total/1e6:.0f} MB across {len(codes)} counties")
