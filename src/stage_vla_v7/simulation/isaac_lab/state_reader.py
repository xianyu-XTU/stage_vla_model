"""Typed Isaac Lab state readers for the V7 physical runtime."""

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
    object_position_w: Tensor
    end_effector_position_w: Tensor
    left_fingertip_position_w: Tensor
    right_fingertip_position_w: Tensor
    finger_a_force_n: Tensor
    finger_b_force_n: Tensor


@dataclass(frozen=True)
class PlacementState:
    object_position_w: Tensor
    support_position_w: Tensor
    object_linear_velocity_w: Tensor
    object_angular_velocity_w: Tensor
    support_linear_velocity_w: Tensor
    support_angular_velocity_w: Tensor
    gripper_joint_position: Tensor
    gripper_open: Tensor


def to_torch(value: object) -> Tensor:
    """Convert Isaac ProxyArray-like values without assuming NumPy support."""
    if hasattr(value, "torch"):
        value = value.torch
    return torch.as_tensor(value)


def resolve_frame_indices(frame_names: Iterable[str]) -> FrameIndices:
    names = list(frame_names)

    def one(name: str) -> int:
        matches = [index for index, value in enumerate(names) if value == name]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one target frame {name!r}; got {names!r}"
            )
        return matches[0]

    return FrameIndices(
        end_effector=one("end_effector"),
        left_tip=one("tool_leftfinger"),
        right_tip=one("tool_rightfinger"),
    )

def frame_positions_w(frame_transformer: object) -> tuple[FrameIndices, Tensor]:
    indices = resolve_frame_indices(frame_transformer.data.target_frame_names)
    positions = to_torch(frame_transformer.data.target_pos_w)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise RuntimeError(f"unexpected target_pos_w shape: {tuple(positions.shape)}")
    return indices, positions


def net_force_per_env(sensor: object) -> Tensor:
    """Reduce one per-finger contact sensor to one magnitude per environment."""
    forces = sensor.data.net_forces_w
    if forces is None:
        raise RuntimeError("contact sensor net_forces_w is None")
    forces = to_torch(forces)
    if forces.ndim < 2 or forces.shape[-1] != 3:
        raise RuntimeError(
            f"expected a final contact-force dimension of 3, got {tuple(forces.shape)}"
        )
    if not torch.isfinite(forces).all():
        raise RuntimeError("contact sensor net_forces_w contains NaN/Inf")
    norms = torch.linalg.vector_norm(forces, dim=-1)
    if norms.ndim == 1:
        return norms
    return norms.reshape(norms.shape[0], -1).amax(dim=-1)


def read_grasp_state(
    base_env: object,
    *,
    object_asset_name: str = "cube_2",
    frame_transformer_name: str = "ee_frame",
    finger_a_sensor_name: str = "left_finger_contact",
    finger_b_sensor_name: str = "right_finger_contact",
) -> GraspState:
    indices, positions = frame_positions_w(base_env.scene[frame_transformer_name])
    object_position = to_torch(
        base_env.scene[object_asset_name].data.root_pos_w
    )[..., :3]
    return GraspState(
        object_position_w=object_position,
        end_effector_position_w=positions[:, indices.end_effector, :3],
        left_fingertip_position_w=positions[:, indices.left_tip, :3],
        right_fingertip_position_w=positions[:, indices.right_tip, :3],
        finger_a_force_n=net_force_per_env(base_env.scene[finger_a_sensor_name]),
        finger_b_force_n=net_force_per_env(base_env.scene[finger_b_sensor_name]),
    )


def read_gripper_joint_positions(base_env: object) -> Tensor:
    """Return actual Franka finger joint positions with shape `[N,2]`."""
    robot = base_env.scene["robot"]
    if not hasattr(base_env.cfg, "gripper_joint_names"):
        raise RuntimeError("environment config has no gripper_joint_names")
    joint_ids, _ = robot.find_joints(base_env.cfg.gripper_joint_names)
    if len(joint_ids) != 2:
        raise RuntimeError(f"expected two gripper joints, got ids={joint_ids}")
    positions = to_torch(robot.data.joint_pos)[:, joint_ids]
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise RuntimeError(f"unexpected gripper joint shape: {tuple(positions.shape)}")
    if not torch.isfinite(positions).all():
        raise RuntimeError("gripper joint positions contain NaN/Inf")
    return positions


def read_placement_state(
    base_env: object,
    *,
    object_asset_name: str = "cube_2",
    support_asset_name: str = "cube_1",
    gripper_open_tolerance_m: float = 0.002,
) -> PlacementState:
    """Read object/support motion and the measured Franka gripper state."""
    if gripper_open_tolerance_m < 0.0:
        raise ValueError("gripper_open_tolerance_m must be non-negative")
    manipulated = base_env.scene[object_asset_name]
    support = base_env.scene[support_asset_name]
    gripper_joint_position = read_gripper_joint_positions(base_env)
    if not hasattr(base_env.cfg, "gripper_open_val"):
        raise RuntimeError("environment config has no gripper_open_val")
    open_target = torch.full_like(
        gripper_joint_position, float(base_env.cfg.gripper_open_val)
    )
    gripper_open = torch.isclose(
        gripper_joint_position,
        open_target,
        atol=gripper_open_tolerance_m,
        rtol=0.0,
    ).all(dim=-1)
    return PlacementState(
        object_position_w=to_torch(manipulated.data.root_pos_w)[..., :3],
        support_position_w=to_torch(support.data.root_pos_w)[..., :3],
        object_linear_velocity_w=to_torch(manipulated.data.root_lin_vel_w)[..., :3],
        object_angular_velocity_w=to_torch(manipulated.data.root_ang_vel_w)[..., :3],
        support_linear_velocity_w=to_torch(support.data.root_lin_vel_w)[..., :3],
        support_angular_velocity_w=to_torch(support.data.root_ang_vel_w)[..., :3],
        gripper_joint_position=gripper_joint_position,
        gripper_open=gripper_open,
    )
