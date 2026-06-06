"""Mean-RGB + texture rules that label a polygon as one of the 6 classes.

★ THIS IS THE FILE YOU EDIT. ★
The rest of the sidecar treats `classify_by_color` as data — change the
thresholds here, hit `Classify` again, watch the map update. No other
code needs to change when you tune these rules.

Two-stage pipeline recap:
  1. SAM 3 finds every interesting polygon in the image (boundary only).
  2. THIS function decides what class each polygon belongs to, given
     the average colour and texture of the pixels inside the polygon.

Class IDs match `prompts.py::LAND_COVER`:
    0 = unclassified  (return when no rule is confident)
    1 = forest
    2 = grassland
    3 = cropland
    4 = water
    5 = bare_soil
    6 = built_up

Tuning workflow (do this in QGIS or Preview):
  1. Open the input GeoTIFF you'll run classify on.
  2. Eye-drop a few pixels you KNOW are forest. Note R, G, B.
  3. Repeat for each class.
  4. Write/adjust rules below so each of those colour points hits the
     right branch.

Rules of thumb:
  - Order matters: the first matching branch wins. Put the rules with
    the most distinctive cues (water, forest) first.
  - Use INEQUALITIES between channels, not absolute thresholds where you
    can avoid it — your imagery's overall luminance shifts with daylight
    and season, but channel RATIOS are more stable.
  - `std_*` is the per-channel pixel standard deviation under the mask.
    High std = textured (forest, urban); low std = uniform (water,
    fresh-paved roads).
"""

from __future__ import annotations


def classify_by_color(
    r: int, g: int, b: int,
    std_r: int, std_g: int, std_b: int,
) -> int:
    """Return a class_id in 0..6 from one polygon's mean colour stats.

    Args are mean and std of the R, G, B channels under the polygon
    mask, all in 0..255 range. Returns 0 (unclassified) when no rule
    matches — those polygons are dropped before writing the GPKG.

    The seed rules below work well enough to verify the pipeline. You
    should override them based on YOUR imagery's colour distribution.
    """
    # ── TODO(human): tune these. ────────────────────────────────────────
    # The conditions below are a starting point chosen to be visually
    # interpretable. They will mis-label many polygons until you adjust.
    # ────────────────────────────────────────────────────────────────────

    # Water: blue clearly dominant + low texture.
    if b > r + 10 and b > g + 5 and std_b < 35:
        return 4  # water

    # Forest: green dominant, mid-to-high luminance, textured.
    if g > r and g > b and (g - max(r, b)) >= 5 and std_g >= 12:
        return 1  # forest

    # Cropland: yellow-brown patches. R > G > B with a wide R–B gap.
    # Bright fields tend to have R 130–180, G 110–160, B 80–120.
    if r > g and g >= b and (r - b) >= 20 and r >= 110:
        return 3  # cropland

    # Built-up: near-gray (channels within ±15) and bright.
    if abs(r - g) <= 18 and abs(g - b) <= 18 and r >= 110:
        return 6  # built_up

    # Bare soil: dim brownish (r > g > b), lower luminance than built-up.
    if r > g > b and (r - b) >= 10 and r < 130:
        return 5  # bare_soil

    # Grassland: yellow-green, lower texture than forest.
    if g > r and g >= b and (g - r) >= 3 and std_g < 18:
        return 2  # grassland

    return 0  # unclassified
