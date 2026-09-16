"""Public boundary between simulator backends and the VLA core."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import (
    RobotAction,
    RobotObservation,
    SimulationAction,
    SimulationModelDescriptor,
    SimulationObservation,
    SimulationState,
)
from .vision_interface import VisionRequest


@runtime_checkable
class SimulationAdapter(Protocol):
    @property
    def descriptor(self) -> SimulationModelDescriptor: ...

    def to_vision_request(self, observation: SimulationObservation) -> VisionRequest: ...

    def to_robot_observation(self, observation: SimulationObservation) -> RobotObservation: ...

    def to_simulation_action(self, action: RobotAction) -> SimulationAction: ...


@runtime_checkable
class SimulationEnvironment(Protocol):
    @property
    def descriptor(self) -> SimulationModelDescriptor: ...

    def reset(self, seed: int | None = None) -> SimulationState: ...

    def observe(self) -> SimulationObservation: ...

    def step(self, action: SimulationAction) -> SimulationState: ...

    def close(self) -> None: ...
