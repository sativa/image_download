"""Build a genuine CROSS-county split from existing data (no new downloads).

All 89 training counties already have S2 + multitemporal-NDVI + DLTB on .250.
We hold out 12 whole counties (county-disjoint) as the test set and train on
the other 77. Test = up to 10 cells per held-out county (~120 cells) — ~15x the
old 8-cell test, and honestly cross-county (the old 8 cells were same-county).

Seeded (42) for reproducibility / paper.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

HOME = Path("/home/ps/landform")
R = json.loads((HOME / "data/v27_regions.json").read_text())

by_county = defaultdict(list)
for r in R["train"]:
    by_county[r["county"]].append(r)
counties = sorted(by_county)
print(f"total training counties: {len(counties)}")

rng = random.Random(42)
test_counties = sorted(rng.sample(counties, 12))
train_counties = [c for c in counties if c not in test_counties]

new_train = [r for c in train_counties for r in by_county[c]]
new_test = []
for c in test_counties:
    cells = list(by_county[c])
    rng.shuffle(cells)
    new_test.extend(cells[:10])

out = {
    "train": new_train,
    "test": new_test,
    "n_train": len(new_train),
    "n_test": len(new_test),
    "train_counties": train_counties,
    "test_counties": test_counties,
    "split": "xcounty_12holdout_seed42",
}
(HOME / "data/v40_xcounty_regions.json").write_text(json.dumps(out, ensure_ascii=False))
print("HELD-OUT test counties (12):", test_counties)
print(f"train: {len(new_train)} cells / {len(train_counties)} counties")
print(f"test:  {len(new_test)} cells / {len(test_counties)} counties")
print("test cells per county:",
      {c: sum(1 for r in new_test if r["county"] == c) for c in test_counties})
