"""Expand semantic task relations into registered skill tokens."""

from __future__ import annotations

from stage_vla_v7.interfaces import SkillToken, TaskPlan

from .action_list import ACTION_REGISTRY, ActionRegistry


class TaskScheduler:
    def __init__(self, registry: ActionRegistry = ACTION_REGISTRY) -> None:
        registry.validate_complete()
        self.registry = registry

    def schedule(self, plan: TaskPlan) -> tuple[SkillToken, ...]:
        return tuple(
            SkillToken(index, skill, relation.object_label, relation.support_label)
            for index, relation in enumerate(plan.execution_relations)
            for skill in self.registry.sequence
        )
