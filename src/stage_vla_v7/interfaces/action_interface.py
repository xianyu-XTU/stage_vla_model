"""Public action boundary with no policy-network or simulator dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Sequence
from typing import Mapping, Protocol, runtime_checkable

from .contracts import ModelDescriptor, ObjectProfile, RobotAction, RobotObservation, Skill


@dataclass(frozen=True)
class ActionRequest:
    skill: Skill
    observation: tuple[float, ...]
    object_profile: ObjectProfile
    support_profile: ObjectProfile
    finished: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = self.observation.values if isinstance(self.observation, RobotObservation) else self.observation
        observation = tuple(float(value) for value in values)
        if not observation or not all(math.isfinite(value) for value in observation):
            raise ValueError("action observation must contain finite values")
        object.__setattr__(self, "observation", observation)


@dataclass(frozen=True)
class ActionResult:
    action: RobotAction
    provider: ModelDescriptor
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class ActionPolicy(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...

    @property
    def observation_dim(self) -> int: ...

    def predict(self, request: ActionRequest) -> ActionResult: ...


class BatchActionPolicy(Protocol):
    def predict_batch(self, requests: Sequence[ActionRequest]) -> tuple[ActionResult, ...]: ...
