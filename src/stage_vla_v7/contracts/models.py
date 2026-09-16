"""Compatibility exports for contracts now owned by ``interfaces``."""

from typing import Mapping

from stage_vla_v7.interfaces.contracts import (
    ModelDescriptor,
    ObjectDetection,
    ObjectProfile,
    RobotAction,
    RobotObservation,
    SKILL_SEQUENCE,
    SceneState,
    SimulationAction,
    SimulationModelDescriptor,
    SimulationObservation,
    SimulationState,
    Skill,
    SkillToken,
    StackRelation,
    TaskPlan,
    expand_skill_tokens,
)

Diagnostics = Mapping[str, object]

__all__ = [
    "Diagnostics",
    "ModelDescriptor",
    "ObjectDetection",
    "ObjectProfile",
    "RobotAction",
    "RobotObservation",
    "SKILL_SEQUENCE",
    "SceneState",
    "SimulationAction",
    "SimulationModelDescriptor",
    "SimulationObservation",
    "SimulationState",
    "Skill",
    "SkillToken",
    "StackRelation",
    "TaskPlan",
    "expand_skill_tokens",
]
