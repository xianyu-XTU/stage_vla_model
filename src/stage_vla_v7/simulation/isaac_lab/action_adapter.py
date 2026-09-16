"""Convert validated V7 actions to Isaac Lab tensor commands."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from stage_vla_v7.interfaces import RobotAction, SimulationAction


class IsaacActionAdapter:
    control_mode = "relative_cartesian_5d"

    def to_simulation_action(self, action: RobotAction) -> SimulationAction:
        if not isinstance(action, RobotAction):
            raise TypeError("action must be a RobotAction")
        return SimulationAction(action, self.control_mode, {"backend": "isaac-lab"})

    def to_tensor(self, actions: Iterable[RobotAction], *, like: Any) -> Any:
        values = tuple(action.values for action in actions)
        if not values:
            raise ValueError("at least one RobotAction is required")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - Isaac Lab supplies torch
            raise RuntimeError("Isaac action conversion requires torch") from exc
        reference = torch.as_tensor(like)
        return torch.tensor(values, dtype=torch.float32, device=reference.device)


def known_size_raw_action(
    policy_action: Any,
    *,
    translation_limit_m: float = 0.005,
    yaw_limit_rad: float = 0.02,
) -> tuple[Any, Any]:
    """Decode normalized ``[dx,dy,dz,dyaw,grip]`` for Isaac's 7-D term."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Isaac Lab supplies torch
        raise RuntimeError("Isaac action conversion requires torch") from exc
    value = torch.as_tensor(policy_action)
    if value.ndim != 2 or value.shape[-1] != 5:
        raise ValueError("known-size action must have shape [N,5]")
    if translation_limit_m <= 0 or yaw_limit_rad <= 0:
        raise ValueError("action limits must be positive")
    if not torch.isfinite(value).all():
        raise ValueError("known-size action contains NaN/Inf")
    unit = value.clamp(-1.0, 1.0)
    raw = torch.zeros((unit.shape[0], 7), device=unit.device, dtype=unit.dtype)
    raw[:, :3] = unit[:, :3] * float(translation_limit_m)
    raw[:, 5] = unit[:, 3] * float(yaw_limit_rad)
    raw[:, 6] = unit[:, 4]
    return raw, unit


def reach_raw_action(policy_action: Any) -> Any:
    """Decode normalized REACH ``[dx,dy,dz,dyaw,grip]`` to seven Isaac actions."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Isaac Lab supplies torch
        raise RuntimeError("Isaac action conversion requires torch") from exc
    action = torch.as_tensor(policy_action, dtype=torch.float32)
    if action.ndim != 2 or action.shape[-1] != 5:
        raise ValueError("REACH policy action must have shape [N,5]")
    unit = action.clamp(-1.0, 1.0)
    raw = torch.zeros((unit.shape[0], 7), device=unit.device, dtype=unit.dtype)
    raw[:, :3] = unit[:, :3] * 0.005
    raw[:, 5] = unit[:, 3] * 0.02
    raw[:, 6] = 1.0
    return raw
