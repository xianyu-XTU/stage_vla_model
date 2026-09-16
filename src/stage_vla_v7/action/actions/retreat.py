"""RETREAT skill definition."""

from stage_vla_v7.interfaces import Skill

from .base import ACTION_PARAMETERS, SkillDefinition

DEFINITION = SkillDefinition(
    7, Skill.RETREAT, __name__, "v5-skill-state55-v1", 55,
    ("end_effector", "object_pose", "support_pose", "gripper_state", "object_velocity"),
    ACTION_PARAMETERS, ("box",), "RETREAT",
    "RELEASE_STABILIZE succeeds without a simulator reset",
    "end effector clears the stack in distance and height",
    "stack becomes invalid or is disturbed during withdrawal",
    None,
)
