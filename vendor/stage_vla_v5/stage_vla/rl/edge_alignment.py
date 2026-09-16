"""Deterministic red-cube edge-alignment controller for M9-A/M9-B.

The learned policy does not choose yaw.  Before each Isaac step, this helper
reads the true fingertip closing axis and red-cube orientation, computes the
shortest yaw correction toward a cube local X/Y axis, and injects only dRz into
the verified pose-relative IK action.

This is deliberately a common low-level controller rather than stage-aware
reward/state.  It uses no temporal history and does not change M4-M8 truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from stage_vla.core import read_float
from stage_vla.envs.state_readers import frame_positions_w, to_torch
from stage_vla.stages.grasp_alignment import (
    EdgeAlignmentCommand,
    edge_alignment_command,
    quat_wxyz_planar_axes,
)

from .action_adapter import expand_m9a_policy_action


@dataclass(frozen=True)
class EdgeAlignmentConfig:
    max_yaw_step_rad: float
    tolerance_deg: float
    rotation_action_scale_rad: float

    def __post_init__(self) -> None:
        if self.max_yaw_step_rad <= 0:
            raise ValueError("max_yaw_step_rad must be > 0")
        if self.tolerance_deg < 0 or self.tolerance_deg >= 45.0:
            raise ValueError("tolerance_deg must satisfy 0 <= tolerance_deg < 45")
        if self.rotation_action_scale_rad <= 0:
            raise ValueError("rotation_action_scale_rad must be > 0")


@dataclass(frozen=True)
class EdgeAlignmentRuntime:
    command: EdgeAlignmentCommand
    yaw_delta_raw: Tensor
    closing_axis_xy: Tensor
    cube_x_axis_xy: Tensor
    cube_y_axis_xy: Tensor


def load_edge_alignment_cfg(config_file: Path) -> EdgeAlignmentConfig:
    return EdgeAlignmentConfig(
        max_yaw_step_rad=read_float(config_file, "m9a_grasp_alignment", "max_yaw_step_rad"),
        tolerance_deg=read_float(config_file, "m9a_grasp_alignment", "tolerance_deg"),
        rotation_action_scale_rad=read_float(
            config_file, "m9a_grasp_alignment", "rotation_action_scale_rad"
        ),
    )


def edge_alignment_runtime(base_env, cfg: EdgeAlignmentConfig) -> EdgeAlignmentRuntime:
    indices, frame_pos = frame_positions_w(base_env.scene["ee_frame"])
    left = frame_pos[:, indices.left_tip, :3]
    right = frame_pos[:, indices.right_tip, :3]
    closing_xy = right[:, :2] - left[:, :2]

    red_quat = to_torch(base_env.scene["cube_2"].data.root_quat_w)[..., :4]
    cube_x_xy, cube_y_xy = quat_wxyz_planar_axes(red_quat)
    command = edge_alignment_command(
        closing_xy,
        cube_x_xy,
        cube_y_xy,
        max_step_rad=cfg.max_yaw_step_rad,
        tolerance_deg=cfg.tolerance_deg,
    )
    yaw_raw = command.bounded_step_rad / float(cfg.rotation_action_scale_rad)
    return EdgeAlignmentRuntime(
        command=command,
        yaw_delta_raw=yaw_raw,
        closing_axis_xy=closing_xy,
        cube_x_axis_xy=cube_x_xy,
        cube_y_axis_xy=cube_y_xy,
    )


def edge_aligned_raw_action(
    base_env,
    policy_actions: Tensor,
    cfg: EdgeAlignmentConfig,
) -> tuple[Tensor, EdgeAlignmentRuntime]:
    runtime = edge_alignment_runtime(base_env, cfg)
    raw = expand_m9a_policy_action(policy_actions, yaw_delta_raw=runtime.yaw_delta_raw)
    return raw, runtime
