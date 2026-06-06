"""Pick diverse training + test regions from the county scan output."""

from __future__ import annotations

import json
import random
from pathlib import Path

GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")
CAND_JSON = Path("/tmp/county_candidates.json")
OUT_JSON = Path("/tmp/v11_regions.json")


def main():
    data = json.loads(CAND_JSON.read_text())
    # Drop counties with no usable cells.
    counties = {code: v for code, v in data.items()
                if v and not (len(v) == 1 and "error" in v[0])}
    print(f"usable counties: {len(counties)}")

    rng = random.Random(0)
    counties_sorted = sorted(counties.items(), key=lambda kv: kv[0])

    # Spread training across many counties; reserve 4 counties for test.
    rng.shuffle(counties_sorted)
    test_counties = counties_sorted[:4]
    train_counties = counties_sorted[4:24]  # 20 counties for training

    train_regions = []
    for code, cells in train_counties:
        cells_sorted = sorted(cells, key=lambda c: -c.get("score", 0))
        top = cells_sorted[:30]
        picks = rng.sample(top, k=min(2, len(top)))
        for k, c in enumerate(picks):
            train_regions.append({
                "county": code,
                "idx": k,
                "bbox": c["bbox"],
                "gdb": str(GDB_ROOT / code / "XYBASE.gdb"),
            })

    test_regions = []
    for code, cells in test_counties:
        cells_sorted = sorted(cells, key=lambda c: -c.get("score", 0))
        top = cells_sorted[:10]
        picks = rng.sample(top, k=min(1, len(top)))
        for k, c in enumerate(picks):
            test_regions.append({
                "county": code,
                "idx": k,
                "bbox": c["bbox"],
                "gdb": str(GDB_ROOT / code / "XYBASE.gdb"),
            })

    out = {
        "train": train_regions,
        "test": test_regions,
        "n_counties": len(train_counties),
        "n_train": len(train_regions),
        "n_test": len(test_regions),
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT_JSON}")
    print(f"  train: {len(train_regions)} regions across {len(train_counties)} counties")
    print(f"  test: {len(test_regions)} regions across {len(test_counties)} HELD-OUT counties")
    # Show breakdown
    print(f"  train counties: {sorted({r['county'] for r in train_regions})}")
    print(f"  test counties:  {sorted({r['county'] for r in test_regions})}")


if __name__ == "__main__":
    main()
