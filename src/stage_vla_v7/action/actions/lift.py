"""LIFT skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    2, Skill.LIFT, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "support_pose", "held", "object_velocity"),
    ACTION_PARAMETERS, ("box",), "LIFT",
    "physical grasp is stable",
    "held object clears the configured lift height",
    "grasp lost or object falls below the entrance height",
    Skill.TRANSPORT,
)
