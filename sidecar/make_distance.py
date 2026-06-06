"""Distance-to-boundary labels (ResUNet-a / BsiNet style) for the parcel delineation distance head:
per cell, distance transform of NON-boundary pixels, normalised to [0,1] (clip 30 px). Distance-map
local maxima = parcel centres -> cleaner watershed seeding for dense fields. Reads c_1m_pbound edges."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

D = Path("/mnt/sda/zf/landform/data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pbound-dir", required=True)   # e.g. c_1m_pbound or c_1m_tibet_pbound
    p.add_argument("--out", required=True)          # e.g. c_1m_dist or c_1m_tibet_dist
    p.add_argument("--clip", type=float, default=30.0)
    a = p.parse_args()
    src = Path(a.pbound_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npy")); n = 0
    for f in files:
        o = out / f.name
        if o.exists():
            n += 1; continue
        edge = np.load(f) > 0
        interior = (~edge).astype(np.uint8)
        dist = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
        dist = np.clip(dist / a.clip, 0, 1).astype(np.float32)   # 0 at edge -> 1 deep inside parcel
        np.save(o, dist); n += 1
        if n % 1000 == 0: print(f"  {n}/{len(files)}", flush=True)
    print(f"[dist] {n} distance maps -> {out}", flush=True)


if __name__ == "__main__":
    main()
