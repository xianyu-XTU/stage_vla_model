"""Final normalized action projection shared by every policy backend."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import RobotAction, Skill


@dataclass(frozen=True)
class SafetyLimits:
    """Normalized command limits applied after model inference."""

    translation: float = 1.0
    yaw: float = 1.0
    grip_min: float = -1.0
    grip_max: float = 1.0

    def __post_init__(self) -> None:
        if self.translation <= 0.0 or self.yaw <= 0.0 or self.grip_min > self.grip_max:
            raise ValueError("invalid action safety limits")


class SafetyProjector:
    """Clamp model output and hold finished skills in a safe state."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    @staticmethod
    def _finished_grip(skill: Skill) -> float:
        return 1.0 if skill in {Skill.REACH, Skill.RELEASE_STABILIZE, Skill.RETREAT} else -1.0

    def project(self, skill: Skill, action: RobotAction, *, finished: bool = False) -> RobotAction:
        limits = self.limits
        if finished:
            grip = min(limits.grip_max, max(limits.grip_min, self._finished_grip(skill)))
            return RobotAction(0.0, 0.0, 0.0, 0.0, grip)
        dx, dy, dz, dyaw, grip = action.values
        return RobotAction(
            min(limits.translation, max(-limits.translation, dx)),
            min(limits.translation, max(-limits.translation, dy)),
            min(limits.translation, max(-limits.translation, dz)),
            min(limits.yaw, max(-limits.yaw, dyaw)),
            min(limits.grip_max, max(limits.grip_min, grip)),
        )
