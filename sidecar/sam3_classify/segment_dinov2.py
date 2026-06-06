"""Stage-1 segmentation via DINOv2 patch embeddings + k-means clustering.

Strategy:
  1. Run DINOv2 once on the whole image to get a (H/14, W/14, 1024) grid
     of patch embeddings.
  2. K-means cluster the embeddings into K=80 groups. Each cluster
     becomes one "segment" — pixels that DINOv2 considers visually
     similar end up in the same group.
  3. Upsample the patch-resolution cluster map to image resolution
     (nearest-neighbour) so each pixel has a cluster ID.
  4. For each cluster ID, materialise its boolean mask and return it
     to Stage 2 as a (score=1.0, mask) tuple.

Trade-offs vs SAM 3:
  + 100% pixel coverage by construction (every pixel belongs to a
    cluster); no "unclassified" output from Stage 1.
  + One forward pass on the whole image — predictable runtime,
    typically faster than SAM 3 grid-point on CPU.
  + No tile-snap or resize artefacts — DINOv2 is fully convolutional
    over the patch grid.
  - Boundaries snap to the 14-px patch grid, so they're blockier than
    SAM 3's masks; the per-class dissolve + simplify steps smooth them
    afterwards.
  - K is a fixed hyperparameter; if your scene has fewer than K visual
    classes, some clusters will be near-duplicates (Stage 2 dissolves
    them by colour, so this is harmless).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_model(weights_dir: Path, device: str):
    """Load DINOv2-large from a local snapshot directory.

    Uses `transformers.AutoModel` instead of torch.hub so we don't need
    network access at runtime. The weights dir must contain
    `config.json` and `model.safetensors`.
    """
    from transformers import AutoImageProcessor, AutoModel
    import torch

    processor = AutoImageProcessor.from_pretrained(str(weights_dir))
    model = AutoModel.from_pretrained(str(weights_dir))
    model = model.to(device)
    model = getattr(model, "eval")()
    return processor, model


def _embed_patches(
    rgb: np.ndarray,
    processor,
    model,
    device: str,
):
    """Run DINOv2 once on the full image; return (Ph, Pw, D) patch grid.

    DINOv2-large uses 14x14 patches; on a 768x1280 input that's a
    54x91 grid, so the inference is one ViT forward pass — fast.
    """
    import torch
    from PIL import Image

    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values=pixel_values, output_hidden_states=False)
    # Last hidden state: (1, 1+N_patches, D). Drop the CLS token.
    last = out.last_hidden_state
    tokens = last[0, 1:, :]  # (N, D)
    # Reshape to (Ph, Pw, D). The processor pads/resizes to a multiple
    # of 14; we can infer Ph, Pw from the model config or count.
    n_patches = tokens.shape[0]
    # transformers DINOv2 processor uses the input height/width to
    # compute the grid; pull them from `inputs["pixel_values"].shape`.
    _, _, H_in, W_in = pixel_values.shape
    patch = 14
    Ph, Pw = H_in // patch, W_in // patch
    assert Ph * Pw == n_patches, f"patch grid {Ph}x{Pw}={Ph*Pw} != {n_patches}"
    return tokens.detach().cpu().numpy().reshape(Ph, Pw, -1), (H_in, W_in)


def auto_segment(
    tif_path: Path,
    n_clusters: int = 80,
    device: str = "cpu",
    dino_weights_dir: str = "/Users/zhangfeng/D/dinov2_weights/dinov2-large",
    on_progress=None,
) -> tuple[list[tuple[float, np.ndarray]], tuple[int, int]]:
    """DINOv2 + k-means segmentation; returns same shape as
    segment_samgeo.auto_segment so the two backends are interchangeable.
    """
    from sklearn.cluster import MiniBatchKMeans
    import rasterio

    if on_progress:
        on_progress(0, 3, stage="loading_model")
    processor, model = _load_model(Path(dino_weights_dir), device)

    if on_progress:
        on_progress(1, 3, stage="reading_image")
    with rasterio.open(tif_path) as src:
        bands = src.read(out_dtype="uint8")
        H, W = src.height, src.width
        if bands.shape[0] < 3:
            raise ValueError("need >=3 bands")
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)

    if on_progress:
        on_progress(2, 3, stage="embedding")
    patch_grid, (H_in, W_in) = _embed_patches(rgb, processor, model, device)
    Ph, Pw, D = patch_grid.shape
    flat = patch_grid.reshape(-1, D)

    # KMeans clusters DINOv2 features into n_clusters visual groups.
    # MiniBatchKMeans is dramatically faster than vanilla KMeans on CPU
    # without meaningful quality loss for our grid size.
    if on_progress:
        on_progress(3, 4, stage="clustering")
    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, flat.shape[0]),
        batch_size=1024,
        random_state=0,
        n_init=3,
        max_iter=200,
    )
    cluster_at_patch = kmeans.fit_predict(flat).reshape(Ph, Pw)

    # Upsample patch labels to image resolution. nearest-neighbour by
    # repeat keeps the assignments crisp at the patch boundary; the
    # post-processing simplify step smooths the resulting steps.
    cluster_full = np.kron(cluster_at_patch, np.ones((14, 14), dtype=np.int32))
    # The processor may have padded to multiples of 14; crop back to the
    # original image's size.
    cluster_full = cluster_full[:H, :W]
    if cluster_full.shape != (H, W):
        # Fallback if crop fails (e.g. heavy padding): resize via PIL.
        from PIL import Image
        cluster_full = np.array(
            Image.fromarray(cluster_full.astype(np.int32)).resize(
                (W, H), Image.NEAREST
            )
        )

    if on_progress:
        on_progress(4, 4, stage="done")

    # Each cluster becomes one "instance" the downstream pipeline can
    # classify. Score is uniform 1.0 since DINOv2 clustering doesn't
    # have a per-mask confidence; the colour rules decide the label.
    instances: list[tuple[float, np.ndarray]] = []
    for c in range(int(cluster_full.max()) + 1):
        m = cluster_full == c
        if not m.any():
            continue
        instances.append((1.0, m.astype(bool)))
    return instances, (H, W)
