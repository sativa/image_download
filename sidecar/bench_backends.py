"""Compare segmentation backends on a single GeoTIFF.

Usage:
    python bench_backends.py <input.tif> [--weights /path/sam3.pt] [--backends sam3,dino,slic]

For each backend it spawns `python -m sam3_classify` as a subprocess
(same code path the Tauri app uses), captures the NDJSON stream, times
the run, and reads the resulting GPKG to produce a comparison table.

Why subprocess instead of just calling `infer.run` in-process?
  - Mirrors production exactly (the Tauri pipeline talks to a Python
    subprocess), so what we benchmark IS what ships.
  - Each backend gets a fresh process so model load times are honest;
    a long-running process would amortise that away.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd


def run_backend(
    python_exe: str,
    sidecar_dir: Path,
    input_tif: Path,
    output_tif: Path,
    weights: Path,
    backend: str,
    confidence: float = 0.05,
) -> dict:
    cmd = [
        python_exe, "-m", "sam3_classify",
        "--input", str(input_tif),
        "--output", str(output_tif),
        "--weights", str(weights),
        "--device", "cpu",
        "--confidence", str(confidence),
        "--backend", backend,
    ]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(sidecar_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ,
             "PYTORCH_ENABLE_MPS_FALLBACK": "1",
             "PYTHONWARNINGS": "ignore"},
    )
    stdout, stderr = proc.communicate()
    elapsed = time.time() - t0
    done_event = None
    last_stage = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "done":
            done_event = rec
        elif rec.get("type") == "stage":
            last_stage = rec.get("stage")
        elif rec.get("type") == "error":
            return {"backend": backend, "ok": False, "error": rec.get("message"),
                    "elapsed": elapsed, "stderr_tail": stderr[-500:]}
    if proc.returncode != 0 or done_event is None:
        return {
            "backend": backend, "ok": False,
            "error": f"exit {proc.returncode}, last stage {last_stage!r}",
            "elapsed": elapsed,
            "stderr_tail": stderr[-500:],
        }

    # Load the GPKG that was written; pixel stats come from the done event.
    gpkg = Path(done_event["label_gpkg"])
    g = gpd.read_file(gpkg, layer="landform")
    stats = done_event.get("stats", {})
    per_class = {}
    for _, row in g.iterrows():
        per_class[row.label] = {
            "area_pct": float(row.area_pct),
            "area_m2": float(row.area_m2),
            "polygon_parts": (
                len(row.geometry.geoms) if row.geometry.geom_type == "MultiPolygon" else 1
            ),
        }
    unclassified_pct = float(stats.get("0", {}).get("area_pct", 0.0))
    return {
        "backend": backend,
        "ok": True,
        "elapsed": elapsed,
        "polygons_dissolved": len(g),
        "per_class": per_class,
        "unclassified_pct": unclassified_pct,
        "gpkg_path": str(gpkg),
        "gpkg_size_kb": gpkg.stat().st_size / 1024,
    }


def format_table(results: list[dict]) -> str:
    """One-line summary per backend + per-class area table."""
    out = []
    out.append(f"{'backend':<8} {'ok':<3} {'time(s)':<10} {'parts':<8} {'unclass%':<10} {'gpkg(KB)':<10}")
    out.append("-" * 60)
    for r in results:
        if not r["ok"]:
            out.append(f"{r['backend']:<8} {'no':<3} {r['elapsed']:<10.1f} -- -- --  {r.get('error','')}")
            continue
        out.append(
            f"{r['backend']:<8} {'yes':<3} {r['elapsed']:<10.1f} "
            f"{r['polygons_dissolved']:<8} {r['unclassified_pct']:<10.2f} "
            f"{r['gpkg_size_kb']:<10.0f}"
        )
    out.append("")
    out.append("Per-class area % (rows: backends, cols: classes):")
    all_labels = sorted({label for r in results if r["ok"] for label in r["per_class"]})
    header = f"{'backend':<8} " + " ".join(f"{l[:10]:<11}" for l in all_labels)
    out.append(header)
    for r in results:
        if not r["ok"]:
            continue
        row = f"{r['backend']:<8} " + " ".join(
            f"{r['per_class'].get(l, {}).get('area_pct', 0):<11.2f}" for l in all_labels
        )
        out.append(row)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_tif", type=Path)
    p.add_argument("--weights", type=Path,
                   default=Path("/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt"))
    p.add_argument("--backends", default="slic,dino,sam3",
                   help="Comma-separated subset of {sam3,dino,slic}")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/bench_landform"))
    p.add_argument("--python",
                   default="/Users/zhangfeng/D/sam3/sam3_env_py312/bin/python")
    p.add_argument("--sidecar-dir",
                   default="/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = []
    for backend in backends:
        print(f"\n=== Running backend: {backend} ===", flush=True)
        out = args.out_dir / f"bench_{backend}.landform.tif"
        # Remove stale outputs from a previous run
        for sib in args.out_dir.glob(f"bench_{backend}.landform.*"):
            sib.unlink()
        r = run_backend(
            args.python, Path(args.sidecar_dir),
            args.input_tif, out, args.weights, backend,
        )
        results.append(r)
        print(json.dumps(r, default=str)[:400], flush=True)

    print("\n" + "=" * 60)
    print(format_table(results))
    print("=" * 60)

    # Persist the raw report alongside the outputs.
    (args.out_dir / "bench_report.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
