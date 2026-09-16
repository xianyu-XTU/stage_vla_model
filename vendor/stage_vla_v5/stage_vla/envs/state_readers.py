"""Centralized Isaac-state readers for M3/M4 diagnostics.

Rules enforced here:
- resolve FrameTransformer indices from ``target_frame_names``;
- treat ``root_pos_w`` / ``target_pos_w`` as world-frame data;
- convert ProxyArray-like values to torch before checking shapes;
- use ``torch.linalg.vector_norm(..., dim=-1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrameIndices:
    end_effector: int
    left_tip: int
    right_tip: int


@dataclass(frozen=True)
class GraspState:
    red_pos_w: Tensor
    ee_pos_w: Tensor
    left_tip_w: Tensor
    right_tip_w: Tensor
    finger_a_force_n: Tensor
    finger_b_force_n: Tensor


@dataclass(frozen=True)
class GraspGeometryState:
    """Positions required by a contact-free pre-CLOSE geometry decision."""

    red_pos_w: Tensor
    left_tip_w: Tensor
    right_tip_w: Tensor


@dataclass(frozen=True)
class PlacementState:
    red_pos_w: Tensor
    blue_pos_w: Tensor
    red_lin_vel_w: Tensor
    red_ang_vel_w: Tensor
    blue_lin_vel_w: Tensor
    blue_ang_vel_w: Tensor
    gripper_joint_pos: Tensor
    gripper_open: Tensor


def to_torch(value) -> Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    return torch.as_tensor(value)


def resolve_frame_indices(frame_names: Iterable[str]) -> FrameIndices:
    names = list(frame_names)

    def one(name: str) -> int:
        matches = [i for i, value in enumerate(names) if value == name]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one target frame {name!r}; got {names!r}"
            )
        return matches[0]

    return FrameIndices(
        end_effector=one("end_effector"),
        left_tip=one("tool_leftfinger"),
        right_tip=one("tool_rightfinger"),
    )


def frame_positions_w(ee_frame) -> tuple[FrameIndices, Tensor]:
    indices = resolve_frame_indices(ee_frame.data.target_frame_names)
    positions = to_torch(ee_frame.data.target_pos_w)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected target_pos_w shape: {tuple(positions.shape)}"
        )
    return indices, positions


def net_force_per_env(sensor) -> Tensor:
    """Reduce one per-finger ContactSensor to one force magnitude per env."""
    forces = sensor.data.net_forces_w
    if forces is None:
        raise RuntimeError("ContactSensor net_forces_w is None.")

    forces = to_torch(forces)
    if forces.ndim < 2 or forces.shape[-1] != 3:
        raise RuntimeError(
            "Expected final contact-force vector dimension 3 after conversion, "
            f"got {tuple(forces.shape)}"
        )
    if not torch.isfinite(forces).all():
        raise RuntimeError("ContactSensor net_forces_w contains NaN/Inf.")

    norms = torch.linalg.vector_norm(forces, dim=-1)
    if norms.ndim == 1:
        return norms
    return norms.reshape(norms.shape[0], -1).amax(dim=-1)



def filtered_force_per_env(sensor) -> Tensor:
    """Reduce filtered ``force_matrix_w`` to one partner-contact magnitude/env.

    Expected Isaac Lab shape is ``[N, B, M, 3]`` (or an equivalent singleton
    variant).  We intentionally use the filtered matrix rather than
    ``net_forces_w`` so support contact with Cube_1 is not confused with finger,
    table, or other contacts on the red cube.
    """
    forces = sensor.data.force_matrix_w
    if forces is None:
        raise RuntimeError(
            "ContactSensor force_matrix_w is None; install it with filter_prim_paths_expr"
        )
    forces = to_torch(forces)
    if forces.ndim < 2 or forces.shape[-1] != 3:
        raise RuntimeError(f"Unexpected filtered contact-force shape: {tuple(forces.shape)}")
    if not torch.isfinite(forces).all():
        raise RuntimeError("ContactSensor force_matrix_w contains NaN/Inf")
    norms = torch.linalg.vector_norm(forces, dim=-1)
    if norms.ndim == 1:
        return norms
    return norms.reshape(norms.shape[0], -1).amax(dim=-1)

def read_grasp_state(
    base_env,
    *,
    red_cube_name: str = "cube_2",
    ee_frame_name: str = "ee_frame",
    finger_a_sensor_name: str = "left_finger_contact",
    finger_b_sensor_name: str = "right_finger_contact",
) -> GraspState:
    ee_frame = base_env.scene[ee_frame_name]
    indices, positions = frame_positions_w(ee_frame)

    red = base_env.scene[red_cube_name]
    red_pos_w = to_torch(red.data.root_pos_w)[..., :3]

    return GraspState(
        red_pos_w=red_pos_w,
        ee_pos_w=positions[:, indices.end_effector, :3],
        left_tip_w=positions[:, indices.left_tip, :3],
        right_tip_w=positions[:, indices.right_tip, :3],
        finger_a_force_n=net_force_per_env(base_env.scene[finger_a_sensor_name]),
        finger_b_force_n=net_force_per_env(base_env.scene[finger_b_sensor_name]),
    )


def read_grasp_geometry_state(
    base_env,
    *,
    red_cube_name: str = "cube_2",
    ee_frame_name: str = "ee_frame",
) -> GraspGeometryState:
    """Read geometry without touching either ContactSensor."""

    ee_frame = base_env.scene[ee_frame_name]
    indices, positions = frame_positions_w(ee_frame)
    red = base_env.scene[red_cube_name]
    red_pos_w = to_torch(red.data.root_pos_w)[..., :3]
    return GraspGeometryState(
        red_pos_w=red_pos_w,
        left_tip_w=positions[:, indices.left_tip, :3],
        right_tip_w=positions[:, indices.right_tip, :3],
    )




def read_gripper_joint_positions(base_env) -> Tensor:
    """Return actual Franka two-finger joint positions, shape ``(num_envs, 2)``."""
    robot = base_env.scene["robot"]
    if not hasattr(base_env.cfg, "gripper_joint_names"):
        raise RuntimeError("Environment cfg has no gripper_joint_names.")

    joint_ids, _ = robot.find_joints(base_env.cfg.gripper_joint_names)
    if len(joint_ids) != 2:
        raise RuntimeError(
            "Expected exactly two parallel-gripper joints, "
            f"got ids={joint_ids}."
        )

    all_joint_pos = to_torch(robot.data.joint_pos)
    gripper_joint_pos = all_joint_pos[:, joint_ids]
    if gripper_joint_pos.ndim != 2 or gripper_joint_pos.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected gripper joint position shape: {tuple(gripper_joint_pos.shape)}"
        )
    if not torch.isfinite(gripper_joint_pos).all():
        raise RuntimeError("Gripper joint positions contain NaN/Inf.")
    return gripper_joint_pos

def read_placement_state(
    base_env,
    *,
    red_cube_name: str = "cube_2",
    blue_cube_name: str = "cube_1",
    gripper_open_tolerance_m: float = 0.002,
) -> PlacementState:
    """Read M7 placement/release state from the vectorized Isaac environment.

    ``gripper_open`` is based on the *actual Franka finger joint positions*,
    not on the last action command. This mirrors the official stack task's
    choice to verify physical gripper state when declaring an object stacked.
    """
    if gripper_open_tolerance_m < 0:
        raise ValueError("gripper_open_tolerance_m must be >= 0")

    red = base_env.scene[red_cube_name]
    blue = base_env.scene[blue_cube_name]
    red_pos_w = to_torch(red.data.root_pos_w)[..., :3]
    blue_pos_w = to_torch(blue.data.root_pos_w)[..., :3]
    red_lin_vel_w = to_torch(red.data.root_lin_vel_w)[..., :3]
    red_ang_vel_w = to_torch(red.data.root_ang_vel_w)[..., :3]
    blue_lin_vel_w = to_torch(blue.data.root_lin_vel_w)[..., :3]
    blue_ang_vel_w = to_torch(blue.data.root_ang_vel_w)[..., :3]

    if not hasattr(base_env.cfg, "gripper_open_val"):
        raise RuntimeError("Environment cfg has no gripper_open_val.")

    gripper_joint_pos = read_gripper_joint_positions(base_env)
    open_target = torch.full_like(
        gripper_joint_pos,
        float(base_env.cfg.gripper_open_val),
    )
    gripper_open = torch.isclose(
        gripper_joint_pos,
        open_target,
        atol=gripper_open_tolerance_m,
        rtol=0.0,
    ).all(dim=-1)

    return PlacementState(
        red_pos_w=red_pos_w,
        blue_pos_w=blue_pos_w,
        red_lin_vel_w=red_lin_vel_w,
        red_ang_vel_w=red_ang_vel_w,
        blue_lin_vel_w=blue_lin_vel_w,
        blue_ang_vel_w=blue_ang_vel_w,
        gripper_joint_pos=gripper_joint_pos,
        gripper_open=gripper_open,
    )
