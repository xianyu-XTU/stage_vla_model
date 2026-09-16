"""Control and rigid-body limits shared by simulation model descriptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlLimits:
    translation_per_step_m: float = 0.005
    yaw_per_step_rad: float = 0.02
    gripper_effort_n: float = 40.0

    def __post_init__(self) -> None:
        if min(
            self.translation_per_step_m,
            self.yaw_per_step_rad,
            self.gripper_effort_n,
        ) <= 0.0:
            raise ValueError("control limits must be positive")


@dataclass(frozen=True)
class RigidBodyLimits:
    max_linear_velocity_mps: float = 1000.0
    max_angular_velocity_radps: float = 1000.0
    max_depenetration_velocity_mps: float = 5.0

    def __post_init__(self) -> None:
        if min(
            self.max_linear_velocity_mps,
            self.max_angular_velocity_radps,
            self.max_depenetration_velocity_mps,
        ) <= 0.0:
            raise ValueError("rigid-body limits must be positive")
