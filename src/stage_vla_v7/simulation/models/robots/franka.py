"""Franka Panda model used by the current Isaac Lab stack environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stage_vla_v7.interfaces import SimulationModelDescriptor

from ...physics import ControlLimits


@dataclass(frozen=True)
class FrankaModel:
    descriptor: SimulationModelDescriptor = SimulationModelDescriptor(
        "franka-panda",
        "robot",
        "isaac-stack-cube-franka-ik-rel-v0",
        "isaac-lab",
        ("relative-ik", "parallel-jaw", "rgbd-stack"),
    )
    asset_reference: str = "Isaac-Stack-Cube-Franka-IK-Rel-v0:scene.robot"
    joint_names: tuple[str, ...] = (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    end_effector_link: str = "panda_hand"
    gripper_links: tuple[str, str] = ("panda_leftfinger", "panda_rightfinger")
    joint_limits_rad_or_m: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "panda_joint1": (-2.8973, 2.8973),
            "panda_joint2": (-1.7628, 1.7628),
            "panda_joint3": (-2.8973, 2.8973),
            "panda_joint4": (-3.0718, -0.0698),
            "panda_joint5": (-2.8973, 2.8973),
            "panda_joint6": (-0.0175, 3.7525),
            "panda_joint7": (-2.8973, 2.8973),
            "panda_finger_joint1": (0.0, 0.04),
            "panda_finger_joint2": (0.0, 0.04),
        }
    )
    default_joint_pose: tuple[float, ...] = (
        0.0,
        -0.569,
        0.0,
        -2.810,
        0.0,
        3.037,
        0.741,
        0.04,
        0.04,
    )
    control_limits: ControlLimits = ControlLimits()
    supported_controllers: tuple[str, ...] = ("differential-ik-relative",)

    def __post_init__(self) -> None:
        if len(self.default_joint_pose) != len(self.joint_names):
            raise ValueError("Franka default pose must match its joint list")
        if set(self.joint_names) != set(self.joint_limits_rad_or_m):
            raise ValueError("Franka joint limits must cover every joint")
