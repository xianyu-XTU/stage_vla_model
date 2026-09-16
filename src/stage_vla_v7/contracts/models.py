"""Dependency-free data contracts shared by vision, language, and action."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence

from .errors import ContractError


def _finite_tuple(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ContractError(f"{name} must contain {size} finite values")
    return result


class Skill(str, Enum):
    """Canonical V7 manipulation skills."""

    REACH = "REACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    ALIGN = "ALIGN"
    DESCEND = "DESCEND"
    RELEASE_STABILIZE = "RELEASE_STABILIZE"
    RETREAT = "RETREAT"


SKILL_SEQUENCE: tuple[Skill, ...] = tuple(Skill)


@dataclass(frozen=True)
class ModelDescriptor:
    """Stable identity and advertised capabilities of one provider."""

    name: str
    version: str
    kind: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "kind"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"descriptor {field_name} must be non-empty")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ContractError("descriptor capabilities must not contain duplicates")


@dataclass(frozen=True)
class ObjectDetection:
    """One object pose estimated in the robot-root coordinate frame."""

    label: str
    position_xyz_m: tuple[float, float, float]
    yaw_rad: float = 0.0
    confidence: float = 1.0
    size_xyz_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ContractError("detection label must be non-empty")
        object.__setattr__(
            self,
            "position_xyz_m",
            _finite_tuple(self.position_xyz_m, 3, "position_xyz_m"),
        )
        if not math.isfinite(float(self.yaw_rad)):
            raise ContractError("yaw_rad must be finite")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ContractError("confidence must be in [0, 1]")
        if self.size_xyz_m is not None:
            size = _finite_tuple(self.size_xyz_m, 3, "size_xyz_m")
            if any(value <= 0.0 for value in size):
                raise ContractError("size_xyz_m values must be positive")
            object.__setattr__(self, "size_xyz_m", size)


@dataclass(frozen=True)
class SceneState:
    """Validated detector output independent of any particular task."""

    detections: tuple[ObjectDetection, ...]
    frame_id: str | None = None
    timestamp_s: float | None = None
    held_label: str | None = None

    def __post_init__(self) -> None:
        detections = tuple(self.detections)
        if not detections:
            raise ContractError("scene must contain at least one detection")
        labels = [item.label for item in detections]
        if len(labels) != len(set(labels)):
            raise ContractError("scene contains duplicate detection labels")
        if self.timestamp_s is not None and not math.isfinite(float(self.timestamp_s)):
            raise ContractError("timestamp_s must be finite")
        if self.held_label is not None and self.held_label not in labels:
            raise ContractError("held_label must name a detected object")
        object.__setattr__(self, "detections", detections)

    def detection(self, label: str) -> ObjectDetection:
        """Return one required detection by stable label."""
        for item in self.detections:
            if item.label == label:
                return item
        raise ContractError(f"scene is missing required detection {label!r}")


@dataclass(frozen=True)
class StackRelation:
    """One semantic instruction to place an object on a support object."""

    object_label: str
    support_label: str

    def __post_init__(self) -> None:
        for name in ("object_label", "support_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} must be non-empty")
        if self.object_label == self.support_label:
            raise ContractError("an object cannot be stacked on itself")


def _execution_order(relations: tuple[StackRelation, ...]) -> tuple[StackRelation, ...]:
    """Order a stack from its bottom relation upward, rejecting cycles."""
    dependencies: dict[int, set[int]] = {index: set() for index in range(len(relations))}
    for index, relation in enumerate(relations):
        for candidate_index, candidate in enumerate(relations):
            if index != candidate_index and relation.support_label == candidate.object_label:
                dependencies[index].add(candidate_index)
    result: list[StackRelation] = []
    remaining = set(range(len(relations)))
    while remaining:
        ready = [index for index in sorted(remaining) if not dependencies[index] & remaining]
        if not ready:
            raise ContractError("stack relations contain a dependency cycle")
        for index in ready:
            result.append(relations[index])
            remaining.remove(index)
    return tuple(result)


@dataclass(frozen=True)
class TaskPlan:
    """Language output containing semantic relations but no robot action."""

    relations: tuple[StackRelation, ...]
    source_text: str = ""

    def __post_init__(self) -> None:
        relations = tuple(self.relations)
        if not relations:
            raise ContractError("task plan must contain at least one relation")
        if len(relations) != len(set(relations)):
            raise ContractError("task plan contains duplicate relations")
        _execution_order(relations)
        object.__setattr__(self, "relations", relations)

    @property
    def execution_relations(self) -> tuple[StackRelation, ...]:
        """Return relations ordered from the base of the stack upward."""
        return _execution_order(self.relations)


@dataclass(frozen=True)
class SkillToken:
    """One action skill bound to one semantic relation."""

    relation_index: int
    skill: Skill
    object_label: str
    support_label: str

    def __post_init__(self) -> None:
        if self.relation_index < 0:
            raise ContractError("relation_index must be non-negative")


def expand_skill_tokens(plan: TaskPlan) -> tuple[SkillToken, ...]:
    """Expand every ordered stack relation into the canonical skill chain."""
    return tuple(
        SkillToken(index, skill, relation.object_label, relation.support_label)
        for index, relation in enumerate(plan.execution_relations)
        for skill in SKILL_SEQUENCE
    )


@dataclass(frozen=True)
class ObjectProfile:
    """Physical metadata resolved outside vision and language models."""

    label: str
    geometry: str
    size_xyz_m: tuple[float, float, float]
    mass_kg: float
    grasp_mode: str = "parallel_jaw"
    support_mode: str = "flat"
    friction_coefficient: float = 0.8
    deformable: bool = False
    stackable: bool = True

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.geometry.strip():
            raise ContractError("object label and geometry must be non-empty")
        size = _finite_tuple(self.size_xyz_m, 3, "size_xyz_m")
        if any(value <= 0.0 for value in size):
            raise ContractError("object size values must be positive")
        object.__setattr__(self, "size_xyz_m", size)
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ContractError("mass_kg must be positive")
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient <= 0.0:
            raise ContractError("friction_coefficient must be positive")


@dataclass(frozen=True)
class RobotAction:
    """Normalized physical command in the frozen V7 action order."""

    dx: float
    dy: float
    dz: float
    dyaw: float
    grip: float

    def __post_init__(self) -> None:
        _finite_tuple(self.values, 5, "robot action")

    @property
    def values(self) -> tuple[float, float, float, float, float]:
        return (self.dx, self.dy, self.dz, self.dyaw, self.grip)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "RobotAction":
        checked = _finite_tuple(values, 5, "robot action")
        return cls(*checked)

    def clipped(self, low: float = -1.0, high: float = 1.0) -> "RobotAction":
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ContractError("invalid action clipping bounds")
        return RobotAction.from_values(min(high, max(low, value)) for value in self.values)


Diagnostics = Mapping[str, object]


@dataclass(frozen=True)
class ProviderResult:
    """Common audit fields that provider-specific result types may embed."""

    provider: ModelDescriptor
    diagnostics: Diagnostics = field(default_factory=dict)
