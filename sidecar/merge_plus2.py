"""Build c_1m_plus2 = symlink-merge of c_1m (5000 train + 120 test) + c_1m_terrace2 (5500 train).
Standard 120 test kept as the test split; the 500 held-out terrace-test cells stay in c_1m_terrace2
for a separate terrace-specific eval. Mirrors how c_1m_plus was assembled for the 7000-cell run.
"""
import json, os
from pathlib import Path

base = Path("/mnt/sda/zf/landform/data")
plus = base / "c_1m_plus2"; plus.mkdir(exist_ok=True)
c1 = json.load(open(base / "c_1m/manifest.json"))
t2 = json.load(open(base / "c_1m_terrace2/manifest.json"))


def link(src_dir, names):
    n = 0
    for name in names:
        s = base / src_dir / f"{name}.npz"; d = plus / f"{name}.npz"
        if s.exists() and not d.exists():
            os.symlink(s, d); n += 1
    return n


a = link("c_1m", c1["train"] + c1["test"])
b = link("c_1m_terrace2", t2["train"])
man = {"train": c1["train"] + t2["train"], "test": c1["test"]}
json.dump(man, open(plus / "manifest.json", "w"))
print(f"c_1m_plus2: linked {a} base + {b} terrace2; manifest train={len(man['train'])} test={len(man['test'])}")
