"""GRASP skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    1, Skill.GRASP, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "finger_tips", "contact_force", "gripper_state"),
    ACTION_PARAMETERS, ("box",), "GRASP",
    "REACH pregrasp geometry remains valid",
    "physical grasp predicate remains true for stable steps",
    "object dropped, contact geometry invalid, or timeout",
    Skill.LIFT,
)
