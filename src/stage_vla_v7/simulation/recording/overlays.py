"""Small, control-independent observer-video overlays."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def draw_overlay(
    frame: np.ndarray,
    *,
    step: int,
    skill: str,
    seed: int | None = None,
    env_id: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return a same-resolution RGB frame with a concise observer overlay."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    prefix = []
    if seed is not None:
        prefix.append(f"seed {seed}")
    if env_id is not None:
        prefix.append(f"env {env_id}")
    prefix.append(f"step {int(step)}")
    top = "  ".join(prefix)
    bottom = str(skill)
    if metadata and metadata.get("status"):
        bottom = f"{bottom}  {metadata['status']}"
    draw.rectangle((0, 0, image.width, 26), fill=(15, 15, 20))
    draw.rectangle((0, max(0, image.height - 26), image.width, image.height), fill=(15, 15, 20))
    draw.text((8, 4), top, font=font, fill=(230, 230, 230))
    draw.text((8, max(0, image.height - 23)), bottom, font=font, fill=(120, 220, 255))
    return np.asarray(image)
