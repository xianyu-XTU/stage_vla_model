"""ALIGN skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    4, Skill.ALIGN, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "support_pose", "held", "object_velocity"),
    ACTION_PARAMETERS, ("box",), "ALIGN",
    "TRANSPORT succeeds above a valid support",
    "held object is in the alignment XY and height band at low speed",
    "grasp lost, object too low, or support distance exceeds the hard limit",
    Skill.DESCEND,
)
