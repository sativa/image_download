"""Land-cover classification sidecar for imagery_downloader.

Reads a GeoTIFF produced by the downloader, runs SAM 3 with a fixed set of
text prompts (one per land-cover class), composes per-pixel class labels by
keeping the highest-score prediction, and writes a single-band uint8
GeoTIFF that preserves the input's geographic transform and CRS.

This package is **not** importable until you patch sam3's environment.
Always import `env_patches` first and call `apply()` before importing sam3.
"""
