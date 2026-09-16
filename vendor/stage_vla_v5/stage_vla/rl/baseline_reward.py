"""Pure-PyTorch M9-A non-stage-aware baseline reward helpers.

M9-A is intentionally a *baseline*: it does not use M8's stage tracker,
previous-potential state, stage transitions, or history-gated success bonus.
It provides standard dense manipulation terms that are active concurrently,
so M10 can later measure the incremental effect of stage-aware reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class M9ABaselineRewardConfig:
    reach_std_m: float
    lift_resting_center_z_m: float
    lift_target_delta_m: float
    goal_gate_lift_delta_m: float
    goal_std_m: float
    goal_fine_std_m: float

    def validate(self) -> None:
        if self.reach_std_m <= 0:
            raise ValueError("reach_std_m must be > 0")
        if not torch.isfinite(torch.tensor(self.lift_resting_center_z_m)):
            raise ValueError("lift_resting_center_z_m must be finite")
        if self.lift_target_delta_m <= 0:
            raise ValueError("lift_target_delta_m must be > 0")
        if self.goal_gate_lift_delta_m < 0:
            raise ValueError("goal_gate_lift_delta_m must be >= 0")
        if self.goal_std_m <= 0:
            raise ValueError("goal_std_m must be > 0")
        if self.goal_fine_std_m <= 0:
            raise ValueError("goal_fine_std_m must be > 0")


def _finite_nonnegative(name: str, value: Tensor) -> Tensor:
    value = torch.as_tensor(value)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if torch.any(value < 0):
        raise ValueError(f"{name} must be non-negative")
    return value


def smooth_distance_reward(distance_m: Tensor, *, std_m: float) -> Tensor:
    """Map a non-negative distance to ``(0, 1]`` using ``1 - tanh(d/std)``."""
    if std_m <= 0:
        raise ValueError("std_m must be > 0")
    distance = _finite_nonnegative("distance_m", distance_m)
    return 1.0 - torch.tanh(distance / float(std_m))


def lift_progress_reward(
    red_center_z_w: Tensor,
    *,
    resting_center_z_m: float,
    target_lift_delta_m: float,
) -> Tensor:
    """Current-state lift progress in ``[0, 1]`` relative to the reset table height.

    This is deliberately not M6's history-aware lift truth.  M9-A is the
    non-stage-aware baseline and therefore uses a simple current-state proxy.
    """
    if target_lift_delta_m <= 0:
        raise ValueError("target_lift_delta_m must be > 0")
    red_z = torch.as_tensor(red_center_z_w)
    if not torch.isfinite(red_z).all():
        raise ValueError("red_center_z_w contains NaN/Inf")
    progress = (red_z - float(resting_center_z_m)) / float(target_lift_delta_m)
    return torch.clamp(progress, min=0.0, max=1.0)


def gate_reward_by_bool(reward: Tensor, gate: Tensor) -> Tensor:
    """Apply a current-frame boolean gate to a dense reward.

    This helper is intentionally stateless: it introduces no stage, history,
    or previous-step dependency.  M9-A uses it to prevent causal leakage such
    as receiving lift/transport reward when the red cube is not physically
    grasped in the current frame.
    """
    value = torch.as_tensor(reward)
    mask = torch.as_tensor(gate, device=value.device)
    if mask.dtype is not torch.bool:
        raise TypeError(f"gate must be torch.bool, got {mask.dtype}")
    if mask.shape != value.shape:
        raise ValueError(
            f"gate shape {tuple(mask.shape)} must match reward shape {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("reward contains NaN/Inf")
    return value * mask.to(value.dtype)



def pregrasp_pose_reward(
    red_pos_w: Tensor,
    left_tip_w: Tensor,
    right_tip_w: Tensor,
    *,
    target_tip_height_offset_m: float,
    xy_std_m: float,
    z_std_m: float,
) -> Tensor:
    """Dense current-frame reward for reaching a physically useful pre-grasp pose.

    The target is expressed with the *true fingertip midpoint* rather than only
    the end-effector origin:

    - midpoint XY should coincide with the red-cube center XY;
    - midpoint Z should sit ``target_tip_height_offset_m`` above the red center.

    The default +1 cm vertical offset comes from the already validated M4
    scripted grasp geometry on this project.  This helper is stateless and does
    not use stage/history information.
    """
    if xy_std_m <= 0 or z_std_m <= 0:
        raise ValueError("xy_std_m and z_std_m must be > 0")
    if not torch.isfinite(torch.tensor(float(target_tip_height_offset_m))):
        raise ValueError("target_tip_height_offset_m must be finite")

    red = torch.as_tensor(red_pos_w)
    left = torch.as_tensor(left_tip_w, device=red.device, dtype=red.dtype)
    right = torch.as_tensor(right_tip_w, device=red.device, dtype=red.dtype)
    if red.shape != left.shape or red.shape != right.shape or red.ndim < 1 or red.shape[-1] != 3:
        raise ValueError("red_pos_w/left_tip_w/right_tip_w must share shape (..., 3)")
    if not torch.isfinite(red).all() or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("pregrasp pose inputs contain NaN/Inf")

    midpoint = 0.5 * (left + right)
    xy_error = torch.linalg.vector_norm(midpoint[..., :2] - red[..., :2], dim=-1)
    target_mid_z = red[..., 2] + float(target_tip_height_offset_m)
    z_error = torch.abs(midpoint[..., 2] - target_mid_z)

    xy_reward = smooth_distance_reward(xy_error, std_m=xy_std_m)
    z_reward = smooth_distance_reward(z_error, std_m=z_std_m)
    # Product means both centering and height must become good; neither can
    # completely substitute for the other.
    return xy_reward * z_reward


def gripper_closedness(
    gripper_joint_pos: Tensor,
    *,
    open_joint_pos_m: float,
    closed_joint_pos_m: float,
) -> Tensor:
    """Return actual two-finger closing progress in [0, 1].

    0 means the joints are at the configured open target and 1 means they are
    at the configured closed target. Values outside the nominal range are
    clamped. This uses *actual joint positions*, not the action command.
    """
    if not open_joint_pos_m > closed_joint_pos_m:
        raise ValueError("open_joint_pos_m must be greater than closed_joint_pos_m")
    joints = torch.as_tensor(gripper_joint_pos)
    if joints.ndim < 1 or joints.shape[-1] != 2:
        raise ValueError(f"gripper_joint_pos must end in two finger joints, got {tuple(joints.shape)}")
    if not torch.isfinite(joints).all():
        raise ValueError("gripper_joint_pos contains NaN/Inf")

    denom = float(open_joint_pos_m - closed_joint_pos_m)
    per_finger = (float(open_joint_pos_m) - joints) / denom
    return torch.clamp(per_finger, min=0.0, max=1.0).mean(dim=-1)


def gripper_coordination_reward(
    closedness: Tensor,
    geometry_ready: Tensor,
    *,
    premature_close_penalty_scale: float,
) -> Tensor:
    """Reward closing only when current grasp geometry is ready.

    ``geometry_ready`` is intentionally a current-frame geometric predicate
    (between fingertips + height alignment), not physical contact, stage, or
    history.  Closing before the geometry is ready receives a small penalty;
    being open while far away receives no positive reward, avoiding a trivial
    "do nothing with open gripper" solution.
    """
    if premature_close_penalty_scale < 0:
        raise ValueError("premature_close_penalty_scale must be >= 0")
    c = torch.as_tensor(closedness)
    ready = torch.as_tensor(geometry_ready, device=c.device)
    if ready.dtype is not torch.bool:
        raise TypeError(f"geometry_ready must be torch.bool, got {ready.dtype}")
    if ready.shape != c.shape:
        raise ValueError(
            f"geometry_ready shape {tuple(ready.shape)} must match closedness shape {tuple(c.shape)}"
        )
    if not torch.isfinite(c).all():
        raise ValueError("closedness contains NaN/Inf")
    if torch.any((c < 0) | (c > 1)):
        raise ValueError("closedness must lie in [0, 1]")

    reward = torch.where(
        ready,
        c,
        -float(premature_close_penalty_scale) * c,
    )
    return reward

def exact_red_on_blue_goal(blue_pos_w: Tensor, *, target_height_diff_m: float) -> Tensor:
    if target_height_diff_m <= 0:
        raise ValueError("target_height_diff_m must be > 0")
    blue = torch.as_tensor(blue_pos_w)
    if blue.ndim < 1 or blue.shape[-1] != 3:
        raise ValueError(f"blue_pos_w must end in xyz, got {tuple(blue.shape)}")
    if not torch.isfinite(blue).all():
        raise ValueError("blue_pos_w contains NaN/Inf")
    offset = torch.zeros_like(blue)
    offset[..., 2] = float(target_height_diff_m)
    return blue + offset


def gated_goal_tracking_reward(
    red_pos_w: Tensor,
    blue_pos_w: Tensor,
    *,
    resting_center_z_m: float,
    gate_lift_delta_m: float,
    target_height_diff_m: float,
    std_m: float,
) -> Tensor:
    """Smooth red-to-stack-goal reward gated by current red lift height.

    The gate is current-state only.  It is not a stage transition or history
    latch and therefore remains a clean M9-A baseline term.
    """
    if gate_lift_delta_m < 0:
        raise ValueError("gate_lift_delta_m must be >= 0")
    red = torch.as_tensor(red_pos_w)
    blue = torch.as_tensor(blue_pos_w, device=red.device, dtype=red.dtype)
    if red.shape != blue.shape or red.ndim < 1 or red.shape[-1] != 3:
        raise ValueError("red_pos_w and blue_pos_w must share xyz shape")
    if not torch.isfinite(red).all() or not torch.isfinite(blue).all():
        raise ValueError("red/blue positions contain NaN/Inf")

    goal = exact_red_on_blue_goal(blue, target_height_diff_m=target_height_diff_m)
    distance = torch.linalg.vector_norm(red - goal, dim=-1)
    dense = smooth_distance_reward(distance, std_m=std_m)
    lifted = (red[..., 2] - float(resting_center_z_m)) >= float(gate_lift_delta_m)
    return dense * lifted.to(dense.dtype)



def postgrasp_blended_target_z(
    blue_center_z_w: Tensor,
    xy_error_m: Tensor,
    *,
    target_height_diff_m: float,
    cube_height_m: float,
    transport_surface_clearance_m: float,
    blend_xy_radius_m: float,
) -> Tensor:
    """Blend from a safe transport height to the final stack height.

    This is a *current-state geometric baseline*, not a stage tracker.  When the
    red cube is far from blue in XY, the preferred red-center height is the
    M7-validated collision-clear transport height.  As XY error shrinks, the
    preferred height smoothly approaches the exact red-on-blue stack height.
    """
    if target_height_diff_m <= 0:
        raise ValueError("target_height_diff_m must be > 0")
    if cube_height_m <= 0:
        raise ValueError("cube_height_m must be > 0")
    if transport_surface_clearance_m < 0:
        raise ValueError("transport_surface_clearance_m must be >= 0")
    if blend_xy_radius_m <= 0:
        raise ValueError("blend_xy_radius_m must be > 0")

    blue_z = torch.as_tensor(blue_center_z_w)
    if not torch.isfinite(blue_z).all():
        raise ValueError("blue_center_z_w contains NaN/Inf")
    xy_error = _finite_nonnegative("xy_error_m", xy_error_m).to(
        device=blue_z.device, dtype=blue_z.dtype
    )
    if xy_error.shape != blue_z.shape:
        raise ValueError(
            f"xy_error_m shape {tuple(xy_error.shape)} must match blue_center_z_w "
            f"shape {tuple(blue_z.shape)}"
        )

    stack_z = blue_z + float(target_height_diff_m)
    # Equal-sized cubes: lower half-height + upper half-height = cube_height.
    safe_transport_z = (
        blue_z + float(cube_height_m) + float(transport_surface_clearance_m)
    )
    alpha = torch.clamp(xy_error / float(blend_xy_radius_m), min=0.0, max=1.0)
    return stack_z + alpha * (safe_transport_z - stack_z)


def postgrasp_pose_reward(
    red_pos_w: Tensor,
    blue_pos_w: Tensor,
    *,
    resting_center_z_m: float,
    gate_lift_delta_m: float,
    target_height_diff_m: float,
    cube_height_m: float,
    transport_surface_clearance_m: float,
    blend_xy_radius_m: float,
    xy_std_m: float,
    z_std_m: float,
) -> Tensor:
    """Dense lift->transport->placement pose shaping, current-frame only.

    The reward becomes active only after the red cube is currently above the
    configured lift gate.  It then rewards two things simultaneously:

    1. reducing red-to-blue XY error;
    2. matching a desired red-center height that is high/safe while far away
       and smoothly descends to the exact stack height near blue.

    No stable-grasp streak, history latch, stage index, transition, or previous
    potential is used; M9-A remains non-stage-aware.
    """
    if gate_lift_delta_m < 0:
        raise ValueError("gate_lift_delta_m must be >= 0")
    if xy_std_m <= 0 or z_std_m <= 0:
        raise ValueError("xy_std_m and z_std_m must be > 0")

    red = torch.as_tensor(red_pos_w)
    blue = torch.as_tensor(blue_pos_w, device=red.device, dtype=red.dtype)
    if red.shape != blue.shape or red.ndim < 1 or red.shape[-1] != 3:
        raise ValueError("red_pos_w and blue_pos_w must share shape (..., 3)")
    if not torch.isfinite(red).all() or not torch.isfinite(blue).all():
        raise ValueError("red/blue positions contain NaN/Inf")

    xy_error = torch.linalg.vector_norm(red[..., :2] - blue[..., :2], dim=-1)
    desired_z = postgrasp_blended_target_z(
        blue[..., 2],
        xy_error,
        target_height_diff_m=target_height_diff_m,
        cube_height_m=cube_height_m,
        transport_surface_clearance_m=transport_surface_clearance_m,
        blend_xy_radius_m=blend_xy_radius_m,
    )
    z_error = torch.abs(red[..., 2] - desired_z)

    xy_score = smooth_distance_reward(xy_error, std_m=xy_std_m)
    z_score = smooth_distance_reward(z_error, std_m=z_std_m)
    lifted = (red[..., 2] - float(resting_center_z_m)) >= float(gate_lift_delta_m)
    return xy_score * z_score * lifted.to(xy_score.dtype)


def placement_ready_mask(
    red_pos_w: Tensor,
    blue_pos_w: Tensor,
    *,
    target_height_diff_m: float,
    xy_tolerance_m: float,
    height_tolerance_m: float,
) -> Tensor:
    """Relaxed current-frame pose gate used only to coordinate release."""
    if target_height_diff_m <= 0:
        raise ValueError("target_height_diff_m must be > 0")
    if xy_tolerance_m <= 0:
        raise ValueError("xy_tolerance_m must be > 0")
    if height_tolerance_m < 0:
        raise ValueError("height_tolerance_m must be >= 0")
    red = torch.as_tensor(red_pos_w)
    blue = torch.as_tensor(blue_pos_w, device=red.device, dtype=red.dtype)
    if red.shape != blue.shape or red.ndim < 1 or red.shape[-1] != 3:
        raise ValueError("red_pos_w and blue_pos_w must share shape (..., 3)")
    if not torch.isfinite(red).all() or not torch.isfinite(blue).all():
        raise ValueError("red/blue positions contain NaN/Inf")

    delta = red - blue
    xy_error = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    height_error = torch.abs(delta[..., 2] - float(target_height_diff_m))
    return (
        (xy_error <= float(xy_tolerance_m))
        & (delta[..., 2] > 0.0)
        & (height_error <= float(height_tolerance_m))
    )


def release_coordination_reward(
    openness: Tensor,
    placement_ready: Tensor,
    carrying: Tensor,
    *,
    premature_open_penalty_scale: float,
) -> Tensor:
    """Teach "keep closed while carrying; open once placement pose is ready".

    - placement-ready + fully closed -> -1
    - placement-ready + fully open   -> +1
    - carrying but not ready         -> small penalty proportional to openness
    - otherwise                      -> 0

    This deliberately uses only current-frame masks and actual joint-derived
    openness.  It is therefore not a stage/history reward.
    """
    if premature_open_penalty_scale < 0:
        raise ValueError("premature_open_penalty_scale must be >= 0")
    o = torch.as_tensor(openness)
    ready = torch.as_tensor(placement_ready, device=o.device)
    carry = torch.as_tensor(carrying, device=o.device)
    if ready.dtype is not torch.bool or carry.dtype is not torch.bool:
        raise TypeError("placement_ready and carrying must be torch.bool")
    if ready.shape != o.shape or carry.shape != o.shape:
        raise ValueError("placement_ready/carrying must match openness shape")
    if not torch.isfinite(o).all():
        raise ValueError("openness contains NaN/Inf")
    if torch.any((o < 0) | (o > 1)):
        raise ValueError("openness must lie in [0, 1]")

    ready_reward = 2.0 * o - 1.0
    premature = -float(premature_open_penalty_scale) * o
    return torch.where(ready, ready_reward, torch.where(carry, premature, torch.zeros_like(o)))

def current_success_reward(
    *,
    geometry_ok: Tensor,
    settled: Tensor,
    gripper_open: Tensor,
    physical_grasp: Tensor,
) -> Tensor:
    """Current-state terminal-quality score without history/stage state.

    Final policy evaluation will still use M7's stricter history-gated success.
    This weaker term is intentionally kept as a baseline reward component.
    """
    masks = [torch.as_tensor(x) for x in (geometry_ok, settled, gripper_open, physical_grasp)]
    if any(x.dtype is not torch.bool for x in masks):
        raise TypeError("success masks must be bool tensors")
    if any(x.shape != masks[0].shape for x in masks[1:]):
        raise ValueError("success masks must share shape")
    return (masks[0] & masks[1] & masks[2] & ~masks[3]).to(torch.float32)
