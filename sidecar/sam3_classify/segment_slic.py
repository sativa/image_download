"""Stage-1 segmentation via SLIC superpixels.

The zero-ML baseline backend. Uses skimage's SLIC implementation to
divide the image into ~N compact superpixels based on color similarity
and spatial proximity. Output is 100% coverage by construction.

Trade-offs:
  + 100% coverage; every pixel is in exactly one superpixel.
  + Fast (~3-10 seconds for a 768x1280 image).
  + No GPU or model dependency.
  - Boundaries are colour-driven, not semantic. A road and an adjacent
    shadow may end up in the same superpixel if their colours overlap.
  - No mask scores; Stage 2 colour rules decide everything.

Useful as a control in the backend benchmark — anything ML-based that
doesn't outperform SLIC by a wide margin is probably not worth the
overhead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def auto_segment(
    tif_path: Path,
    n_segments: int = 400,
    compactness: float = 10.0,
    on_progress=None,
    **_ignored,
) -> tuple[list[tuple[float, np.ndarray]], tuple[int, int]]:
    """SLIC superpixel segmentation.

    Returns the same (instances, hw) tuple shape as the other backends.
    """
    import rasterio
    from skimage.segmentation import slic

    if on_progress:
        on_progress(0, 2, stage="reading_image")
    with rasterio.open(tif_path) as src:
        bands = src.read(out_dtype="uint8")
        H, W = src.height, src.width
        if bands.shape[0] < 3:
            raise ValueError("need >=3 bands")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)

    if on_progress:
        on_progress(1, 2, stage="slic")
    labels = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )

    instances: list[tuple[float, np.ndarray]] = []
    for c in range(int(labels.max()) + 1):
        m = labels == c
        if not m.any():
            continue
        instances.append((1.0, m.astype(bool)))

    if on_progress:
        on_progress(2, 2, stage="done")
    return instances, (H, W)
