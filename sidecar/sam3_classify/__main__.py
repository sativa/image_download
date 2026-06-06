"""CLI entry point for the SAM 3 land-cover sidecar.

Usage (called by Rust):

    python -m sam3_classify \
        --input  /path/to/imagery.tif \
        --output /path/to/imagery.landform.tif \
        --weights /path/to/sam3.pt \
        --device auto

NDJSON is written to stdout (one JSON record per line). Free-form logging
and Python tracebacks go to stderr so they don't pollute the IPC channel.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .infer import InferConfig, run


def _emit_err(message: str) -> None:
    sys.stdout.write(json.dumps({"type": "error", "message": message}) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sam3_classify")
    p.add_argument("--input", required=True, type=Path,
                   help="Input GeoTIFF produced by imagery_downloader.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output single-band landform GeoTIFF.")
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to sam3.pt checkpoint.")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cpu", "mps", "cuda"),
                   help="auto: pick cuda > mps > cpu.")
    p.add_argument("--confidence", type=float, default=0.4,
                   help="SAM 3 confidence threshold passed to the processor.")
    p.add_argument("--backend", default="cropland",
                   choices=("cropland", "parcel_dist", "parcel_bh", "parcel", "landcover", "sam3", "dino", "slic"),
                   help="cropland=binary cropland (DEFAULT); parcel_dist=BEST dist-peak watershed + 7-class "
                        "(GeoParquet); parcel_bh=boundary-head watershed; parcel=SAM3+cropland; "
                        "landcover=7-class; sam3/dino/slic=legacy.")
    p.add_argument("--backbone-dir", type=Path,
                   default=Path("/Users/zhangfeng/D/cropland_dino/dinov3-vitl16-sat493m"),
                   help="DINOv3-Sat backbone dir (cropland/landcover/parcel backends).")
    p.add_argument("--sam3-weights", type=Path,
                   default=Path("/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt"),
                   help="SAM 3 checkpoint (parcel backend only).")
    p.add_argument("--classifier", default="color",
                   choices=("color", "siglip"),
                   help="Stage-2 labeller (legacy backends only). color=RGB rules; siglip=VLM.")
    args = p.parse_args(argv)

    required = [args.input, args.weights]
    if args.backend in ("cropland", "landcover", "parcel", "parcel_dist", "parcel_bh"):
        required.append(args.backbone_dir)
    if args.backend == "parcel":
        required.append(args.sam3_weights)
    for r in required:
        if not r.exists():
            _emit_err(f"file not found: {r}")
            return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = InferConfig(
        input_tif=args.input,
        output_tif=args.output,
        device=None if args.device == "auto" else args.device,  # type: ignore[arg-type]
        weights=args.weights,
        confidence_threshold=args.confidence,
        backend=args.backend,
        classifier=args.classifier,
        backbone_dir=args.backbone_dir,
        sam3_weights=args.sam3_weights,
    )
    try:
        run(cfg)
        return 0
    except KeyboardInterrupt:
        _emit_err("cancelled by signal")
        return 130
    except Exception as exc:
        # Surface a compact one-liner via NDJSON (machine-readable) and the
        # full traceback via stderr (human-readable; the Rust side logs it).
        traceback.print_exc(file=sys.stderr)
        _emit_err(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
