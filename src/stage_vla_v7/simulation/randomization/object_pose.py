"""Non-overlapping tabletop object-pose randomization."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .seeds import seeded_random


def sample_object_positions(
    labels: Sequence[str],
    *,
    seed: int | None,
    x_range_m: tuple[float, float] = (0.40, 0.58),
    y_range_m: tuple[float, float] = (-0.15, 0.15),
    height_m: float = 0.0203,
    minimum_separation_m: float = 0.09,
    max_attempts_per_object: int = 1000,
) -> dict[str, tuple[float, float, float]]:
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("object labels must be non-empty and unique")
    if x_range_m[0] >= x_range_m[1] or y_range_m[0] >= y_range_m[1]:
        raise ValueError("object randomization ranges are invalid")
    if height_m <= 0.0 or minimum_separation_m <= 0.0:
        raise ValueError("object height and separation must be positive")
    randomizer = seeded_random(seed)
    result: dict[str, tuple[float, float, float]] = {}
    for label in labels:
        for _attempt in range(max_attempts_per_object):
            candidate = (
                randomizer.uniform(*x_range_m),
                randomizer.uniform(*y_range_m),
                height_m,
            )
            if all(
                math.hypot(candidate[0] - other[0], candidate[1] - other[1])
                >= minimum_separation_m
                for other in result.values()
            ):
                result[label] = candidate
                break
        else:
            raise RuntimeError("unable to sample a non-overlapping object layout")
    return result
