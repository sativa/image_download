"""Stage-2 classifier: zero-shot SigLIP per-segment classification.

Drops `classify_by_color`'s hand-tuned thresholds and uses a pretrained
vision-language model instead. For each segment we crop the image to
the segment's bounding box (with non-mask pixels blacked out so the
crop doesn't carry surrounding context), encode it with SigLIP, and
take the cosine-argmax against the 6 class text prompts.

Why SigLIP instead of plain CLIP:
  - Sigmoid loss (vs softmax) trained on 4B image-text pairs ⇒ better
    zero-shot multi-label / fine-grained discrimination.
  - "google/siglip-base-patch16-256" is ~370 MB, runs on CPU at ~30 ms
    per crop; ~400 segments per scene = ~12 s extra over color rules.

When to prefer this over color rules:
  - Scenes with strong visual class signature but ambiguous mean colour
    (e.g. dark cropland vs bare soil with similar luminance).
  - Open-vocab extension — easy to add new classes by editing the
    `prompts` list, no threshold re-tuning.

When color rules still win:
  - Pure colour signals (water = blue, snow = white). Adds latency and
    can introduce semantic confusion (SigLIP knows water can be "blue
    swimming pool" or "river" — sometimes mismatches).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


SIGLIP_DEFAULT_DIR = "/Users/zhangfeng/D/siglip_weights/siglip-base-patch16-256"


def _build_class_prompts(land_cover_classes):
    """Map our LAND_COVER tuple → SigLIP-friendly natural-language prompts.

    SigLIP scoring is very sensitive to phrase wording — adding the
    'an aerial photo of' prefix anchors the model in our viewing
    perspective and lifts accuracy a measurable amount.
    """
    out = []
    for c in land_cover_classes:
        # Pick the most natural human-readable phrasing for each class.
        phrase_map = {
            "forest": "a forest seen from above",
            "grassland": "open grassland from above",
            "cropland": "a farmland field from above",
            "water": "a river or lake from above",
            "bare_soil": "exposed bare soil from above",
            "built_up": "buildings and roads from above",
        }
        phrase = phrase_map.get(c.label, f"a {c.label.replace('_', ' ')} from above")
        out.append((c.id, c.label, f"an aerial photo of {phrase}"))
    return out


def _load_model(weights_dir: str, device: str):
    """Lazy-import + load SigLIP. The transformers library is already a
    dependency for DINOv2 so no extra install."""
    from transformers import AutoProcessor, AutoModel
    import torch

    model = AutoModel.from_pretrained(weights_dir)
    processor = AutoProcessor.from_pretrained(weights_dir)
    model = model.to(device)
    model = getattr(model, "eval")()
    return processor, model


def _crop_segment(rgb: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray | None:
    """Crop the image to the mask's bounding box and zero out non-mask
    pixels. Returns None if the segment is empty or too small.

    Padding adds a small margin so the SigLIP processor (which resizes
    to 256x256) has a few extra pixels of context — anchoring "is this
    cropland or grass" partly relies on field-boundary visibility.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = max(0, int(ys.min()) - pad), int(ys.max()) + 1 + pad
    x0, x1 = max(0, int(xs.min()) - pad), int(xs.max()) + 1 + pad
    y1 = min(rgb.shape[0], y1)
    x1 = min(rgb.shape[1], x1)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    crop = rgb[y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1]
    # Black out non-mask pixels so the crop's content is dominated by
    # the segment itself, not surrounding land. SigLIP handles black
    # fill better than NaN/random pad.
    crop[~sub_mask] = 0
    return crop


def classify_instances(
    rgb: np.ndarray,
    instances: list[tuple[float, np.ndarray]],
    land_cover_classes,
    device: str = "cpu",
    weights_dir: str = SIGLIP_DEFAULT_DIR,
    batch_size: int = 32,
    on_progress=None,
) -> list[int]:
    """Return one class_id per input instance, in input order.

    Batches segment crops through SigLIP to amortise tokenization
    overhead. The text-side encoding only happens once.
    """
    import torch
    from PIL import Image

    processor, model = _load_model(weights_dir, device)
    prompts = _build_class_prompts(land_cover_classes)
    class_ids = [p[0] for p in prompts]
    text_inputs = processor(
        text=[p[2] for p in prompts],
        return_tensors="pt", padding="max_length", truncation=True,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        # transformers 5.x SiglipModel.get_text_features returns a
        # BaseModelOutputWithPooling; the actual embedding is in
        # `.pooler_output`. Older releases returned a bare tensor —
        # handle both shapes.
        text_out = model.get_text_features(**text_inputs)
        text_features = getattr(text_out, "pooler_output", text_out)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    out_classes: list[int] = []
    batch_imgs: list[Image.Image] = []
    batch_targets: list[int] = []  # output index that this batch entry maps to

    def _flush():
        if not batch_imgs:
            return
        img_inputs = processor(images=batch_imgs, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            image_out = model.get_image_features(**img_inputs)
            image_features = getattr(image_out, "pooler_output", image_out)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T  # (B, num_classes)
            picks = logits.argmax(dim=-1).cpu().tolist()
        for tgt, pick in zip(batch_targets, picks):
            out_classes[tgt] = class_ids[pick]
        batch_imgs.clear()
        batch_targets.clear()

    for i, (_score, mask) in enumerate(instances):
        out_classes.append(0)  # placeholder, may be overwritten below
        crop = _crop_segment(rgb, mask)
        if crop is None:
            continue  # leaves class 0; downstream fill takes care of it
        batch_imgs.append(Image.fromarray(crop))
        batch_targets.append(i)
        if len(batch_imgs) >= batch_size:
            _flush()
            if on_progress:
                on_progress(i + 1, len(instances))
    _flush()
    if on_progress:
        on_progress(len(instances), len(instances))
    return out_classes
