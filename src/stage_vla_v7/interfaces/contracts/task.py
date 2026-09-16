"""Validated task plan contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ContractError
from .instruction import StackRelation


def execution_order(relations: tuple[StackRelation, ...]) -> tuple[StackRelation, ...]:
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
        execution_order(relations)
        object.__setattr__(self, "relations", relations)

    @property
    def execution_relations(self) -> tuple[StackRelation, ...]:
        return execution_order(self.relations)
