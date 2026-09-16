"""Canonical registry for every executable Stage VLA skill."""

from __future__ import annotations

from collections.abc import Iterable

from stage_vla_v7.interfaces import SKILL_SEQUENCE, Skill

from .actions import ALL_DEFINITIONS
from .actions.base import SkillDefinition


class ActionRegistry:
    """Fail-closed registry for skill definitions and their policy slots."""

    def __init__(self, definitions: Iterable[SkillDefinition] = ()) -> None:
        self._definitions: dict[Skill, SkillDefinition] = {}
        self._ids: dict[int, Skill] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: SkillDefinition, *, replace: bool = False) -> None:
        if not isinstance(definition, SkillDefinition):
            raise TypeError("action registry accepts SkillDefinition values")
        if definition.skill in self._definitions and not replace:
            raise ValueError(f"skill {definition.skill.value!r} is already registered")
        owner = self._ids.get(definition.skill_id)
        if owner is not None and owner is not definition.skill:
            raise ValueError(f"skill id {definition.skill_id} is already registered")
        self._definitions[definition.skill] = definition
        self._ids[definition.skill_id] = definition.skill

    def require(self, skill: Skill | str) -> SkillDefinition:
        try:
            return self._definitions[Skill(skill)]
        except (KeyError, ValueError) as exc:
            raise LookupError(f"unknown or unregistered skill {skill!r}") from exc

    @property
    def sequence(self) -> tuple[Skill, ...]:
        return tuple(
            definition.skill
            for definition in sorted(self._definitions.values(), key=lambda item: item.skill_id)
        )

    def validate_complete(self) -> None:
        if self.sequence != SKILL_SEQUENCE:
            raise ValueError("action registry does not match the frozen V7 skill sequence")

    @property
    def definitions(self) -> tuple[SkillDefinition, ...]:
        return tuple(self.require(skill) for skill in self.sequence)


ACTION_REGISTRY = ActionRegistry(ALL_DEFINITIONS)
ACTION_REGISTRY.validate_complete()


def get_action_definition(skill: Skill | str) -> SkillDefinition:
    return ACTION_REGISTRY.require(skill)
