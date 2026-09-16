"""Tensor-safe expansion of one-environment Isaac scene snapshots."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


def _map_tensors(value: Any, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    return deepcopy(value)


def expand_single_env_state(
    scene_state: dict, n: int, device: str | torch.device
) -> dict:
    """Expand a single-environment scene snapshot to exactly ``n`` clones."""
    if n <= 0:
        raise ValueError("n must be > 0")

    def _expand(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device)
        if tensor.ndim == 0:
            return tensor.clone()
        if tensor.shape[0] != 1:
            raise ValueError(
                "snapshot tensors must have batch dimension 1; "
                f"got shape {tuple(tensor.shape)}"
            )
        repetitions = [n] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repetitions).clone()

    return _map_tensors(scene_state, _expand)
