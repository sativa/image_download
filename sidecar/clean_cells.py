"""Score c_1m training cells by DLTB-edge / image-edge AGREEMENT (clean digitization vs mismatch) and
build c_1m_clean = the top-quality cells (symlinks + manifest), for focused boundary training. No model
needed — pure geometry/gradient, fast."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2

D = Path("/mnt/sda/zf/landform/data")


def score(nm, data, pbound):
    try:
        x6 = np.load(data / f"{nm}.npz")["x6"]; tb = np.load(pbound / f"{nm}.npy") > 0
    except Exception:
        return 0.0
    if not tb.any():
        return 0.0
    gray = cv2.cvtColor(np.ascontiguousarray(x6[:3].transpose(1, 2, 0)), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    return float(grad[tb].mean() / (grad.mean() + 1e-6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(D / "c_1m_lc"))
    p.add_argument("--pbound-dir", default=str(D / "c_1m_pbound"))
    p.add_argument("--out", default=str(D / "c_1m_clean"))
    p.add_argument("--frac", type=float, default=0.5, help="keep top fraction by agreement")
    a = p.parse_args()
    data = Path(a.data_dir); pb = Path(a.pbound_dir); out = Path(a.out); out.mkdir(exist_ok=True)
    man = json.loads((data / "manifest.json").read_text())
    tr = [n for n in man["train"] if (pb / f"{n}.npy").exists()]
    print(f"[clean] scoring {len(tr)} train cells ...", flush=True)
    sc = []
    for i, n in enumerate(tr):
        sc.append((n, score(n, data, pb)))
        if (i + 1) % 1000 == 0: print(f"  {i+1}/{len(tr)}", flush=True)
    sc.sort(key=lambda x: -x[1])
    k = int(len(sc) * a.frac)
    clean = [n for n, _ in sc[:k]]
    for n in clean + man["test"]:
        s = data / f"{n}.npz"; o = out / f"{n}.npz"
        if s.exists() and not o.exists(): os.symlink(s, o)
    json.dump({"train": clean, "test": man["test"]}, open(out / "manifest.json", "w"))
    qs = np.array([s for _, s in sc])
    print(f"[clean] kept top {k}/{len(sc)} (agreement>={sc[k-1][1]:.2f}); "
          f"clean median {np.median([s for _,s in sc[:k]]):.2f} vs dropped {np.median([s for _,s in sc[k:]]):.2f}", flush=True)
    print(f"  c_1m_clean -> train {len(clean)} test {len(man['test'])}", flush=True)


if __name__ == "__main__":
    main()
