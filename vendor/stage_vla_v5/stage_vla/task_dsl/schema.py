"""Typed task and skill tokens for the v5 closed manipulation grammar."""

from __future__ import annotations

from dataclasses import dataclass

SKILL_SEQUENCE: tuple[str, ...] = (
    "REACH",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "ALIGN",
    "DESCEND",
    "RELEASE_STABILIZE",
    "RETREAT",
)


@dataclass(frozen=True)
class TaskSpec:
    """A fully resolved closed-domain stack command."""

    object_name: str
    target_name: str

    def __post_init__(self) -> None:
        for field_name, value in (("object_name", self.object_name), ("target_name", self.target_name)):
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
        if self.object_name == self.target_name:
            raise ValueError("object_name and target_name must differ")


@dataclass(frozen=True)
class StackChainSpec:
    """A multi-stage stack plan expressed as object-on-support relations.

    Relations are kept in the user-specified order, while ``execution_tasks``
    exposes the dependency-safe bottom-up order used by the scheduler.
    """

    relations: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        if not self.relations:
            raise ValueError("relations must contain at least one stack stage")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("duplicate stack relation")
        objects = [relation.object_name for relation in self.relations]
        if len(set(objects)) != len(objects):
            raise ValueError("each object may be moved only once in a stack chain")
        pending = list(self.relations)
        completed_objects: set[str] = set()
        ordered: list[TaskSpec] = []
        while pending:
            ready = [
                relation for relation in pending
                if relation.target_name not in {item.object_name for item in pending}
                or relation.target_name in completed_objects
            ]
            if not ready:
                raise ValueError("stack relations contain a dependency cycle")
            relation = ready[0]
            pending.remove(relation)
            ordered.append(relation)
            completed_objects.add(relation.object_name)
        object.__setattr__(self, "_execution_tasks", tuple(ordered))

    @property
    def execution_tasks(self) -> tuple[TaskSpec, ...]:
        """Return stages in bottom-up order, preserving relation dependencies."""
        return self._execution_tasks


@dataclass(frozen=True)
class SkillToken:
    """One scheduler token; it contains no continuous action."""

    skill: str
    object_name: str
    target_name: str

    def __post_init__(self) -> None:
        if self.skill not in SKILL_SEQUENCE:
            raise ValueError(f"unknown skill: {self.skill}")

    def as_dict(self) -> dict[str, str]:
        return {"skill": self.skill, "object": self.object_name, "target": self.target_name}
