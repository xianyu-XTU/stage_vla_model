"""RELEASE_STABILIZE skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    6, Skill.RELEASE_STABILIZE, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "support_pose", "gripper_state", "object_velocity"),
    ACTION_PARAMETERS, ("box",), "RELEASE_STABILIZE",
    "DESCEND succeeds without a simulator reset",
    "stack geometry is valid while the gripper is open and the object is stable",
    "stack breaks, object moves outside the release domain, or timeout",
    Skill.RETREAT,
)
