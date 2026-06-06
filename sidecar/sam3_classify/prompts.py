"""Land-cover prompt taxonomy and palette.

★ This is the file you (the human) tweak to control what the model looks
  for. The rest of the sidecar treats it as data — change a phrase here
  and the entire pipeline picks it up on the next run.

Conventions:
  - Class ID 0 is reserved for "unclassified" (pixels no prompt fires on).
  - IDs must be contiguous starting from 1 — they become raster values in
    the output GeoTIFF and the legend.json keys.
  - `prompts` is a list of SAM 3 text phrases. They are tried in order;
    the per-pixel winner is whichever phrase produces the highest score
    at that pixel (see infer.py::compose_masks). Listing 2–3 paraphrases
    per class noticeably improves recall — SAM 3 is sensitive to wording.
  - `rgb` is only used by downstream visualisers; the GeoTIFF itself only
    stores the class ID.

The six-class scheme below matches the user's downstream DNDC / soil
workflow (matches the standard Chinese land-cover bins).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LandCoverClass:
    id: int
    label: str
    prompts: tuple[str, ...]
    rgb: tuple[int, int, int]


# ── TODO(human): tune these prompts. ────────────────────────────────────
# What's worth changing here:
#   * Add or drop paraphrases inside each tuple (5–10 lines total to edit).
#   * The phrase order inside a class doesn't matter — they're OR-ed.
#   * Avoid generic words ("area", "region") — they bleed across classes.
#   * If a class is consistently confused with another, write a phrase
#     that contrasts: "bare soil, not pavement".
# ────────────────────────────────────────────────────────────────────────
LAND_COVER: tuple[LandCoverClass, ...] = (
    LandCoverClass(
        id=1, label="forest",
        prompts=("forest", "dense trees", "woodland canopy"),
        rgb=(34, 139, 34),
    ),
    LandCoverClass(
        id=2, label="grassland",
        prompts=("grassland", "meadow", "pasture"),
        rgb=(124, 252, 0),
    ),
    LandCoverClass(
        id=3, label="cropland",
        prompts=("cropland", "farmland", "agricultural field"),
        rgb=(218, 165, 32),
    ),
    LandCoverClass(
        id=4, label="water",
        prompts=("water", "river", "lake"),
        rgb=(30, 144, 255),
    ),
    LandCoverClass(
        id=5, label="bare_soil",
        prompts=("bare soil", "exposed earth", "ploughed soil"),
        rgb=(160, 82, 45),
    ),
    LandCoverClass(
        id=6, label="built_up",
        prompts=("built-up area", "buildings", "urban construction"),
        rgb=(128, 128, 128),
    ),
)


UNCLASSIFIED_RGB = (0, 0, 0)


def all_prompts_flat() -> list[tuple[int, str]]:
    """Flatten the taxonomy into [(class_id, phrase), …] for iteration."""
    out: list[tuple[int, str]] = []
    for cls in LAND_COVER:
        for p in cls.prompts:
            out.append((cls.id, p))
    return out


def legend_dict() -> dict:
    """Serialisable legend with class metadata (no per-image stats)."""
    classes = {
        "0": {"label": "unclassified", "rgb": list(UNCLASSIFIED_RGB)},
    }
    for cls in LAND_COVER:
        classes[str(cls.id)] = {
            "label": cls.label,
            "rgb": list(cls.rgb),
            "prompts": list(cls.prompts),
        }
    return {"classes": classes}
