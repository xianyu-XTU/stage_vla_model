"""Debug-only access to simulator object positions for vision diagnosis."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def read_debug_oracle_local_positions(
    raw: Any,
    vision_origins: Any,
    asset_name: str,
    *,
    to_torch: Callable[[Any], Any],
) -> np.ndarray:
    """Read local simulator truth only for explicit ``debug_oracle`` runs."""
    oracle_world = to_torch(
        raw.unwrapped.scene[asset_name].data.root_pos_w
    )[..., :3]
    return (oracle_world - vision_origins).detach().cpu().numpy()


__all__ = ["read_debug_oracle_local_positions"]
