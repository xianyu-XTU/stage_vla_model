"""DESCEND skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    5, Skill.DESCEND, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "support_pose", "held", "object_velocity"),
    ACTION_PARAMETERS, ("box",), "DESCEND",
    "ALIGN succeeds without a simulator reset",
    "held object meets stack XY/Z tolerances at low speed",
    "grasp lost, object too low, or XY error exceeds the hard limit",
    Skill.RELEASE_STABILIZE,
)
