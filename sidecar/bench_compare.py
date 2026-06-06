"""Automated cross-backend comparison.

For one or more input TIFs, runs each backend, then computes
quantitative comparison metrics that don't need ground truth:

  - Pixel-level pairwise agreement matrix
  - Per-class IoU between backends
  - Cohen's kappa (chance-corrected agreement)
  - Boundary alignment score: how well each backend's class
    boundaries follow image-level Sobel edges (a proxy for "the
    backend respects what the image actually shows")
  - Per-class polygon-part counts pre/post dissolve (lower = cleaner)
  - Speed

Outputs a JSON report and side-by-side preview PNGs in --out-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PYTHON = "/Users/zhangfeng/D/sam3/sam3_env_py312/bin/python"
SIDECAR_DIR = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar"
WEIGHTS = "/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt"


def run_one(input_tif: Path, backend: str, out_dir: Path) -> dict:
    """Run a single backend, return done event + path metadata."""
    output_tif = out_dir / f"{input_tif.stem}__{backend}.landform.tif"
    # Wipe previous artefacts so we don't read a stale GPKG.
    for sib in out_dir.glob(f"{input_tif.stem}__{backend}.landform.*"):
        sib.unlink()
    cmd = [
        PYTHON, "-m", "sam3_classify",
        "--input", str(input_tif),
        "--output", str(output_tif),
        "--weights", WEIGHTS,
        "--device", "cpu",
        "--confidence", "0.05",
        "--backend", backend,
    ]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=SIDECAR_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONWARNINGS": "ignore"},
    )
    stdout, stderr = proc.communicate()
    elapsed = time.time() - t0
    done = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                if rec.get("type") == "done":
                    done = rec
            except json.JSONDecodeError:
                pass
    if done is None:
        return {"backend": backend, "ok": False, "elapsed": elapsed,
                "stderr_tail": stderr[-500:]}
    return {"backend": backend, "ok": True, "elapsed": elapsed, **done}


def rasterize_gpkg(gpkg_path: Path, ref_tif: Path) -> np.ndarray:
    """Rasterise the .gpkg back into a class-id grid on the ref TIF's pixel
    grid. Lets us compute pixel-level metrics between backends.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(ref_tif) as src:
        H, W = src.height, src.width
        transform = src.transform
    g = gpd.read_file(gpkg_path, layer="landform")
    shapes = [
        (geom, int(cid))
        for geom, cid in zip(g.geometry, g["class_id"])
    ]
    raster = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return raster


def pairwise_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixel-by-pixel agreement metrics between two label rasters."""
    assert a.shape == b.shape
    eq = a == b
    overall = float(eq.mean())
    classes = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
    classes = [c for c in classes if c != 0]
    per_class_iou: dict[int, float] = {}
    for c in classes:
        inter = int(((a == c) & (b == c)).sum())
        union = int(((a == c) | (b == c)).sum())
        per_class_iou[int(c)] = inter / union if union else 0.0
    # Cohen's kappa
    n = int(a.size)
    p_a = float(eq.sum()) / n
    # Marginals
    unique_all = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
    p_e = 0.0
    for c in unique_all:
        p_e += (float((a == c).sum()) / n) * (float((b == c).sum()) / n)
    kappa = (p_a - p_e) / (1.0 - p_e) if p_e < 1.0 else 0.0
    return {
        "overall_agreement": overall,
        "cohens_kappa": kappa,
        "per_class_iou": per_class_iou,
        "mean_per_class_iou": float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0,
    }


def boundary_alignment_score(label: np.ndarray, rgb: np.ndarray) -> float:
    """How well a label raster's class-boundaries align with image edges.

    Proxy: take a Sobel magnitude on the grayscale image, take the
    class-boundary mask (4-neighbour different-label edges), and return
    the mean Sobel magnitude UNDER the boundary divided by the overall
    mean Sobel magnitude. >1 means class boundaries land on visually
    edge-rich pixels (good); ~1 means random; <1 means worse than random.
    """
    from scipy import ndimage

    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    sx = ndimage.sobel(gray, axis=0)
    sy = ndimage.sobel(gray, axis=1)
    sobel = np.hypot(sx, sy)
    # Class boundary mask: a pixel is on a boundary if any 4-neighbour
    # has a different class id.
    h, w = label.shape
    boundary = np.zeros_like(label, dtype=bool)
    boundary[1:, :] |= label[1:, :] != label[:-1, :]
    boundary[:-1, :] |= label[:-1, :] != label[1:, :]
    boundary[:, 1:] |= label[:, 1:] != label[:, :-1]
    boundary[:, :-1] |= label[:, :-1] != label[:, 1:]
    if not boundary.any():
        return 0.0
    on_boundary_mean = float(sobel[boundary].mean())
    overall_mean = float(sobel.mean()) + 1e-9
    return on_boundary_mean / overall_mean


def render_overlay_png(rgb: np.ndarray, label: np.ndarray, palette: dict, out_png: Path) -> None:
    """50/50 blend of rgb under a coloured class raster, saved as PNG."""
    from PIL import Image
    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb_tuple in palette.items():
        m = label == cid
        if not m.any():
            continue
        color[m] = rgb_tuple
    blend = (rgb.astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5).astype(np.uint8)
    Image.fromarray(blend).save(out_png, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--backends", default="dino,sam3,slic")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/bench_compare"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",")]
    # Pull the palette from the sidecar's prompts so PNGs match the in-app
    # colour scheme. UNCLASSIFIED gets black; everything else from LAND_COVER.
    sys.path.insert(0, SIDECAR_DIR)
    from sam3_classify.prompts import LAND_COVER, UNCLASSIFIED_RGB
    palette = {0: UNCLASSIFIED_RGB}
    for c in LAND_COVER:
        palette[c.id] = c.rgb

    # Build the full job list — every (scene, backend) pair runs in
    # parallel as an independent subprocess. On M3 Ultra (32 cores) this
    # is comfortably under the core budget for 12+ jobs.
    valid_inputs = [p for p in args.inputs if p.exists()]
    missing = [p for p in args.inputs if not p.exists()]
    for m in missing:
        print(f"[SKIP] {m} not found", flush=True)

    jobs = [(tif, backend) for tif in valid_inputs for backend in backends]
    print(f"Spawning {len(jobs)} jobs in parallel "
          f"({len(valid_inputs)} scenes × {len(backends)} backends)...", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # max_workers = len(jobs) — let the OS scheduler handle contention.
    # Each subprocess is single-threaded torch CPU, so 12 in parallel
    # uses ~12 cores. The M3 Ultra has 24 performance cores, so we have
    # generous headroom.
    job_results: dict[tuple[str, str], dict] = {}
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        future_to_job = {
            ex.submit(run_one, tif, backend, args.out_dir): (tif, backend)
            for tif, backend in jobs
        }
        for fut in as_completed(future_to_job):
            tif, backend = future_to_job[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"backend": backend, "ok": False, "error": f"crash: {e}"}
            job_results[(str(tif), backend)] = r
            tag = "ok" if r.get("ok") else "FAIL"
            print(f"  [{tag}] {tif.name} / {backend} — "
                  f"{r.get('elapsed', 0):.1f}s "
                  f"({len(job_results)}/{len(jobs)} done)", flush=True)
    print(f"\nAll jobs finished in {time.time() - t_start:.1f}s wall.", flush=True)

    # Post-process each scene now that all subprocesses are done.
    overall_report = {"scenes": []}
    for input_tif in valid_inputs:
        # Read input RGB once per scene.
        import rasterio
        with rasterio.open(input_tif) as src:
            bands = src.read(out_dtype="uint8")
        if bands.shape[0] < 3:
            print(f"[SKIP] {input_tif.name}: <3 bands")
            continue
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)

        per_backend: dict[str, dict] = {}
        labels: dict[str, np.ndarray] = {}
        for backend in backends:
            r = job_results.get((str(input_tif), backend))
            if r is None or not r.get("ok"):
                continue
            try:
                lbl = rasterize_gpkg(Path(r["label_gpkg"]), input_tif)
            except Exception as e:
                print(f"  {backend} ({input_tif.name}): rasterize failed: {e}")
                continue
            labels[backend] = lbl
            r["boundary_alignment"] = boundary_alignment_score(lbl, rgb)
            r["unique_classes"] = sorted(set(int(c) for c in np.unique(lbl).tolist()))
            per_backend[backend] = r
            render_overlay_png(
                rgb, lbl, palette,
                args.out_dir / f"{input_tif.stem}__{backend}.preview.png",
            )

        pairs = {}
        names = list(labels.keys())
        for i, a in enumerate(names):
            for b in names[i+1:]:
                pairs[f"{a}_vs_{b}"] = pairwise_agreement(labels[a], labels[b])

        scene = {
            "input": str(input_tif),
            "shape": list(rgb.shape[:2]),
            "per_backend": {k: {kk: vv for kk, vv in v.items()
                                 if kk not in ("stats",)} for k, v in per_backend.items()},
            "pairwise": pairs,
        }
        overall_report["scenes"].append(scene)

        print(f"\n── {input_tif.name} ──")
        for b in names:
            r = per_backend[b]
            print(f"    {b:6}: time={r['elapsed']:.1f}s  align={r['boundary_alignment']:.3f}  classes={r['unique_classes']}")
        for k, v in pairs.items():
            print(f"    {k}: kappa={v['cohens_kappa']:.3f}  mean_iou={v['mean_per_class_iou']:.3f}")

    # Cross-scene aggregate.
    agg = {}
    for backend in backends:
        scenes = [s for s in overall_report["scenes"] if backend in s["per_backend"]]
        if not scenes:
            continue
        agg[backend] = {
            "avg_time": float(np.mean([s["per_backend"][backend]["elapsed"] for s in scenes])),
            "avg_boundary_align": float(np.mean([s["per_backend"][backend]["boundary_alignment"] for s in scenes])),
            "ok_rate": sum(1 for s in scenes if s["per_backend"][backend].get("ok")) / len(scenes),
        }
    overall_report["aggregate"] = agg

    # Print ranking.
    print("\n" + "=" * 70)
    print("CROSS-SCENE AGGREGATE:")
    print(f"  {'backend':<8} {'avg_time(s)':<14} {'avg_align':<12} {'ok_rate':<10}")
    print("  " + "-" * 50)
    # Rank by a simple composite: higher align is better, lower time is better.
    ranked = sorted(
        agg.items(),
        key=lambda kv: (-kv[1]["avg_boundary_align"], kv[1]["avg_time"]),
    )
    for name, m in ranked:
        print(f"  {name:<8} {m['avg_time']:<14.1f} {m['avg_boundary_align']:<12.3f} {m['ok_rate']:<10.2f}")
    print("=" * 70)
    if ranked:
        winner = ranked[0][0]
        print(f"\nRECOMMENDED BACKEND: {winner}")
        print(f"  (highest avg boundary alignment, tie-break by speed)")
    print("\nReport saved to:", args.out_dir / "report.json")
    print("Preview PNGs saved to:", args.out_dir)

    (args.out_dir / "report.json").write_text(json.dumps(overall_report, indent=2, default=str))


if __name__ == "__main__":
    main()
