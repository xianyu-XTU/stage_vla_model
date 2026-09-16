"""Deterministic seed handling for reproducible scene resets."""

from __future__ import annotations

import random


def seeded_random(seed: int | None) -> random.Random:
    if seed is not None and seed < 0:
        raise ValueError("simulation seed must be non-negative")
    return random.Random(seed)
