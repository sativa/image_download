"""Pick 80 train regions across ≥40 counties, disjoint from balanced test."""

from __future__ import annotations

import json
import random
from pathlib import Path

GDB_ROOT = Path("/Volumes/Thunderbolt3/三普数据/三调最终成果-20211214")
CAND = Path("/tmp/county_candidates.json")
BAL = Path("/tmp/v11_regions_balanced.json")
OUT = Path("/tmp/v15_regions.json")
N_TRAIN = 80
N_TRAIN_COUNTIES = 40


def main():
    data = json.loads(CAND.read_text())
    balanced = json.loads(BAL.read_text())
    test_counties = {r["county"] for r in balanced["test"]}
    print(f"reserved test counties (8): {sorted(test_counties)}")

    # Usable train candidates: counties with ≥4-class cells, NOT in test set.
    usable = []
    for code, cells in data.items():
        if code in test_counties:
            continue
        if not cells or (cells and "error" in cells[0]):
            continue
        usable.append((code, cells))
    print(f"usable non-test counties: {len(usable)}")

    rng = random.Random(0)
    rng.shuffle(usable)
    # Sample 40 counties × 2 regions each = 80 train regions.
    chosen_counties = usable[:N_TRAIN_COUNTIES]
    train_regions = []
    for code, cells in chosen_counties:
        cells_sorted = sorted(cells, key=lambda c: -c.get("score", 0))
        top = cells_sorted[:30]
        picks = rng.sample(top, k=min(N_TRAIN // N_TRAIN_COUNTIES, len(top)))
        for k, c in enumerate(picks):
            train_regions.append({
                "county": code,
                "idx": k,
                "bbox": c["bbox"],
                "gdb": str(GDB_ROOT / code / "XYBASE.gdb"),
            })
    print(f"selected {len(train_regions)} train regions across {len(chosen_counties)} counties")
    print(f"  county codes: {sorted({r['county'] for r in train_regions})}")

    out = {
        "train": train_regions,
        "test": balanced["test"],  # reuse the same balanced 8-cell test
        "n_train": len(train_regions),
        "n_test": len(balanced["test"]),
        "n_train_counties": len(chosen_counties),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
