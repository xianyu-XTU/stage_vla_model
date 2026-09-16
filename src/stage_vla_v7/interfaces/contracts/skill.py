"""Canonical skill identifiers and task bindings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ContractError
from .task import TaskPlan


class Skill(str, Enum):
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
    """Compatibility scheduler using the frozen canonical skill order."""
    return tuple(
        SkillToken(index, skill, relation.object_label, relation.support_label)
        for index, relation in enumerate(plan.execution_relations)
        for skill in SKILL_SEQUENCE
    )
