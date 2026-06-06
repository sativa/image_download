"""Download Esri+Google imagery for every v11 region in parallel."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
from train_v6_multisource import download_region_source

OUT_DIR = Path("/tmp/v11_imagery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

regions = json.loads(Path("/tmp/v11_regions.json").read_text())
all_regions = regions["train"] + regions["test"]

session = requests.Session()
session.headers["User-Agent"] = "v11/1.0"


jobs = []
for r in all_regions:
    for src in ["esri", "google"]:
        path = OUT_DIR / f"{r['county']}_{r['idx']}_{src}.tif"
        if path.exists():
            continue
        jobs.append((tuple(r["bbox"]), src, path))

print(f"downloading {len(jobs)} images in parallel ({len(all_regions)} regions × 2 sources)")


def _dl(args_):
    bb, src, path = args_
    try:
        download_region_source(bb, 17, src, path, session)
        return path, True
    except Exception as e:
        return path, str(e)[:200]


t0 = time.time()
with ThreadPoolExecutor(max_workers=24) as ex:
    done = 0
    for path, status in ex.map(_dl, jobs):
        done += 1
        if status is True:
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {path.name}")
        else:
            print(f"  [FAIL] {path.name}: {status}")
print(f"done in {time.time()-t0:.1f}s")
print(f"total size: {sum(p.stat().st_size for p in OUT_DIR.glob('*.tif'))/1024/1024:.0f} MB")
