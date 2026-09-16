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
