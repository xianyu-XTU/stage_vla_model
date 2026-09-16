"""Stable scene labels and colors used by whole-task evaluation."""

EXTRA_CUBE_COLORS = (
    (220, 190, 40),
    (170, 80, 210),
    (40, 190, 190),
    (235, 120, 40),
)

ASSET_TO_VISION_LABEL = {
    "cube_1": "blue_cube",
    "cube_2": "red_cube",
    "cube_3": "green_cube",
    "cube_4": "yellow_cube",
}


__all__ = ["ASSET_TO_VISION_LABEL", "EXTRA_CUBE_COLORS"]
