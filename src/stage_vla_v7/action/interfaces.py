"""Action port definitions. This module has no vision or language dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, Sequence, runtime_checkable

from stage_vla_v7.contracts import ModelDescriptor, ObjectProfile, RobotAction, Skill


@dataclass(frozen=True)
class ActionRequest:
    """One policy query with explicit skill and physical roles."""

    skill: Skill
    observation: tuple[float, ...]
    object_profile: ObjectProfile
    support_profile: ObjectProfile
    finished: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation = tuple(float(value) for value in self.observation)
        if not observation or not all(math.isfinite(value) for value in observation):
            raise ValueError("action observation must contain finite values")
        object.__setattr__(self, "observation", observation)


@dataclass(frozen=True)
class ActionResult:
    """One normalized action plus provider audit information."""

    action: RobotAction
    provider: ModelDescriptor
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class ActionPolicy(Protocol):
    """Replaceable state-to-action model boundary."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    @property
    def observation_dim(self) -> int: ...

    def predict(self, request: ActionRequest) -> ActionResult:
        """Return one normalized action for the request skill."""


class BatchActionPolicy(Protocol):
    """Optional high-throughput extension for vectorized Isaac Lab environments."""

    def predict_batch(self, requests: Sequence[ActionRequest]) -> tuple[ActionResult, ...]: ...
