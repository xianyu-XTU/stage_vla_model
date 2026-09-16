"""Simulator-neutral observation, action, state, and model contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from ..errors import ContractError
from .action import RobotAction
from .observation import RobotObservation


@dataclass(frozen=True)
class SimulationModelDescriptor:
    identifier: str
    model_type: str
    version: str
    backend: str = "generic"
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("identifier", "model_type", "version", "backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"simulation model {name} must be non-empty")
        capabilities = tuple(self.capabilities)
        if len(capabilities) != len(set(capabilities)):
            raise ContractError("simulation model capabilities must be unique")
        object.__setattr__(self, "capabilities", capabilities)


@dataclass(frozen=True)
class SimulationObservation:
    robot: RobotObservation
    rgb: object
    depth_m: object | None = None
    frame_id: str | None = None
    timestamp_s: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rgb is None:
            raise ContractError("simulation RGB observation is required")
        if self.timestamp_s is not None and not math.isfinite(float(self.timestamp_s)):
            raise ContractError("simulation timestamp must be finite")


@dataclass(frozen=True)
class SimulationAction:
    robot_action: RobotAction
    control_mode: str = "relative_cartesian_5d"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.control_mode.strip():
            raise ContractError("simulation control mode must be non-empty")


@dataclass(frozen=True)
class SimulationState:
    episode_id: str
    step_index: int
    observation: SimulationObservation
    terminated: bool = False
    truncated: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ContractError("episode_id must be non-empty")
        if self.step_index < 0:
            raise ContractError("step_index must be non-negative")
