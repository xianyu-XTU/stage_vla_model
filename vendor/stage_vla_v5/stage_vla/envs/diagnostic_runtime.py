"""Pure timing helpers for Isaac scripted diagnostics."""

from __future__ import annotations

import math


def episode_step_capacity(
    episode_length_s: float,
    sim_dt_s: float,
    decimation: int,
) -> int:
    if episode_length_s <= 0:
        raise ValueError("episode_length_s must be > 0")
    if sim_dt_s <= 0:
        raise ValueError("sim_dt_s must be > 0")
    if decimation < 1:
        raise ValueError("decimation must be >= 1")
    return math.ceil(episode_length_s / (sim_dt_s * decimation))
