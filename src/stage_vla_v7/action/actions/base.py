"""Metadata contracts used by the canonical action registry."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import Skill


@dataclass(frozen=True)
class ActionParameterDefinition:
    name: str
    minimum: float
    maximum: float
    meaning: str


ACTION_PARAMETERS = (
    ActionParameterDefinition("dx", -1.0, 1.0, "normalized relative Cartesian x"),
    ActionParameterDefinition("dy", -1.0, 1.0, "normalized relative Cartesian y"),
    ActionParameterDefinition("dz", -1.0, 1.0, "normalized relative Cartesian z"),
    ActionParameterDefinition("dyaw", -1.0, 1.0, "normalized relative yaw"),
    ActionParameterDefinition("grip", -1.0, 1.0, "positive opens; negative closes"),
)


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: int
    skill: Skill
    implementation: str
    observation_schema: str
    legacy_observation_dim: int
    required_observation: tuple[str, ...]
    parameter_schema: tuple[ActionParameterDefinition, ...]
    supported_object_domain: tuple[str, ...]
    policy_slot: str
    entrance_condition: str
    success_condition: str
    failure_condition: str
    next_skill: Skill | None

    def __post_init__(self) -> None:
        if self.skill_id < 0 or self.legacy_observation_dim < 1:
            raise ValueError("skill id and observation dimension must be valid")
        if len(self.parameter_schema) != 5:
            raise ValueError("every V7 skill must retain the five-dimensional action schema")
        if self.policy_slot != self.skill.value:
            raise ValueError("policy slot must match the canonical skill name")
