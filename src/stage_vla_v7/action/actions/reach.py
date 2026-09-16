"""REACH skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    0, Skill.REACH, __name__, "v5-reach-state52-v1", 52,
    ("end_effector", "object_pose", "support_pose", "finger_tips", "gripper_state"),
    ACTION_PARAMETERS, ("box",), "REACH",
    "finite object pose within the reachable workspace",
    "open fingertip midpoint aligned above the object for stable steps",
    "invalid pose or end-effector distance beyond the hard workspace bound",
    Skill.GRASP,
)
