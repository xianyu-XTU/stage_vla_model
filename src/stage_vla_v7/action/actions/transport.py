"""TRANSPORT skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    3, Skill.TRANSPORT, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "support_pose", "held", "object_velocity"),
    ACTION_PARAMETERS, ("box",), "TRANSPORT",
    "LIFT succeeds while the object remains held",
    "object is near support in XY with bounded linear and angular speed",
    "grasp lost, object falls, or support distance exceeds the hard limit",
    Skill.ALIGN,
)
