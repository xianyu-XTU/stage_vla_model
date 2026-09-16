"""Pure helpers for safe scripted IK-relative translation commands.

Isaac Lab's DifferentialInverseKinematicsAction performs:

    processed_action = raw_action * cfg.scale

For the Franka stack IK-relative task the official scale is 0.5.

This module works in *metric processed EE delta* first, then converts that
delta back to the raw action expected by the action term. This prevents a
script from confusing world-space target error (meters) with raw action units.

The vector is norm-limited, not component-wise clamped. Component-wise clamp
can produce a diagonal step up to sqrt(3) times larger than the nominal limit.
"""

from __future__ import annotations

import torch
from torch import Tensor


def object_space_servo_target(
    current_ee_pos_w: Tensor,
    current_object_pos_w: Tensor,
    desired_object_pos_w: Tensor,
    *,
    axis_mask: Tensor | tuple[bool, bool, bool] = (True, True, True),
) -> tuple[Tensor, Tensor]:
    """Convert current object-goal error into a receding-horizon EE target.

    A grasped rigid object follows the end effector, but the EE/object offset
    can drift under contact and IK tracking.  Recomputing ``current_ee +
    (desired_object-current_object)`` every control step avoids relying on a
    stale offset captured at one seed-specific pose.  ``axis_mask`` supports
    the vertical-clearance and horizontal-only phases.
    """
    ee = torch.as_tensor(current_ee_pos_w)
    obj = torch.as_tensor(current_object_pos_w, device=ee.device, dtype=ee.dtype)
    desired = torch.as_tensor(desired_object_pos_w, device=ee.device, dtype=ee.dtype)
    if ee.shape != obj.shape or ee.shape != desired.shape or ee.shape[-1] != 3:
        raise ValueError(
            "EE/current-object/desired-object must share (...,3) shape; got "
            f"{tuple(ee.shape)}, {tuple(obj.shape)}, {tuple(desired.shape)}"
        )
    if not torch.isfinite(ee).all() or not torch.isfinite(obj).all() or not torch.isfinite(desired).all():
        raise ValueError("object-space servo inputs contain NaN/Inf")
    mask = torch.as_tensor(axis_mask, device=ee.device, dtype=torch.bool)
    if mask.shape != (3,):
        raise ValueError(f"axis_mask must have shape (3,), got {tuple(mask.shape)}")
    correction = torch.where(mask, desired - obj, torch.zeros_like(obj))
    return ee + correction, correction


def adaptive_step_budget(
    initial_error_m: float,
    max_processed_step_m: float,
    *,
    configured_minimum: int,
    safety_factor: float = 3.0,
) -> int:
    """Return a distance-aware upper bound for randomized scripted motion."""
    if initial_error_m < 0.0:
        raise ValueError("initial_error_m must be >= 0")
    if max_processed_step_m <= 0.0:
        raise ValueError("max_processed_step_m must be > 0")
    if configured_minimum <= 0:
        raise ValueError("configured_minimum must be > 0")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1")
    distance_steps = int(torch.ceil(torch.tensor(initial_error_m / max_processed_step_m)).item())
    return max(configured_minimum, int(torch.ceil(torch.tensor(distance_steps * safety_factor)).item()))


def safer_planar_axis_order(
    current_object_xy_w: Tensor,
    desired_object_xy_w: Tensor,
    robot_root_xy_w: Tensor,
) -> tuple[int, int]:
    """Choose X/Y waypoint order with the smaller intermediate reach radius.

    A direct diagonal Cartesian path can drive the fixed-orientation Franka
    close to a singular or joint-limit configuration.  Both axis orders end at
    the same goal; this deterministic choice minimizes the intermediate planar
    distance from the robot root.  ``(0,1)`` means X then Y.
    """
    current = torch.as_tensor(current_object_xy_w)
    desired = torch.as_tensor(desired_object_xy_w, device=current.device, dtype=current.dtype)
    root = torch.as_tensor(robot_root_xy_w, device=current.device, dtype=current.dtype)
    if current.shape != (2,) or desired.shape != (2,) or root.shape != (2,):
        raise ValueError("planar axis-order inputs must each have shape (2,)")
    if not torch.isfinite(current).all() or not torch.isfinite(desired).all() or not torch.isfinite(root).all():
        raise ValueError("planar axis-order inputs contain NaN/Inf")
    x_first_mid = torch.stack((desired[0], current[1]))
    y_first_mid = torch.stack((current[0], desired[1]))
    x_radius = torch.linalg.vector_norm(x_first_mid - root)
    y_radius = torch.linalg.vector_norm(y_first_mid - root)
    return (0, 1) if float(x_radius.item()) <= float(y_radius.item()) else (1, 0)


def xy_tolerance_gate(
    current_object_pos_w: Tensor,
    desired_object_pos_w: Tensor,
    *,
    tolerance_m: float,
) -> tuple[Tensor, Tensor]:
    """Return planar error and whether it is inside an unchanged XY gate."""
    if tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be > 0")
    current = torch.as_tensor(current_object_pos_w)
    desired = torch.as_tensor(desired_object_pos_w, device=current.device, dtype=current.dtype)
    if current.shape != desired.shape or current.ndim < 1 or current.shape[-1] < 2:
        raise ValueError(
            "current/desired object positions must share a shape ending in at least XY"
        )
    if not torch.isfinite(current).all() or not torch.isfinite(desired).all():
        raise ValueError("XY gate inputs contain NaN/Inf")
    error = torch.linalg.vector_norm(desired[..., :2] - current[..., :2], dim=-1)
    return error, error <= float(tolerance_m)


def limit_vector_norm(vector: Tensor, max_norm: float, *, eps: float = 1.0e-9) -> Tensor:
    """Limit the final-dimension Euclidean norm without changing direction."""
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")
    value = torch.as_tensor(vector)
    if value.ndim < 1:
        raise ValueError("vector must have at least one dimension")
    if not torch.isfinite(value).all():
        raise ValueError("vector contains NaN/Inf")

    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=eps), max=1.0)
    return value * scale


def processed_delta_to_raw_action(
    processed_delta_m: Tensor,
    action_scale: float | Tensor,
) -> Tensor:
    """Convert a metric IK-relative position delta to the raw action units."""
    delta = torch.as_tensor(processed_delta_m)
    scale = torch.as_tensor(action_scale, device=delta.device, dtype=delta.dtype)

    if scale.ndim == 0:
        if float(scale.item()) == 0.0:
            raise ValueError("action_scale must be non-zero")
        return delta / scale

    if scale.shape[-1] not in (1, delta.shape[-1]):
        raise ValueError(
            f"action_scale final dim must be 1 or {delta.shape[-1]}, "
            f"got {tuple(scale.shape)}"
        )
    if torch.any(scale == 0):
        raise ValueError("action_scale contains zero")
    return delta / scale


def raw_ik_relative_position_action(
    target_pos_w: Tensor,
    current_pos_w: Tensor,
    *,
    action_scale: float | Tensor,
    max_processed_step_m: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return raw xyz action, metric processed step, and full target error.

    The caller is responsible for ensuring target/current position deltas are in
    the same frame as the IK relative command. In the current Franka stack task
    the fixed robot base is world-aligned, which is the already-validated project
    diagnostic setup.
    """
    target = torch.as_tensor(target_pos_w)
    current = torch.as_tensor(current_pos_w, device=target.device, dtype=target.dtype)
    if target.shape != current.shape or target.shape[-1] != 3:
        raise ValueError(
            f"target/current must share (...,3) shape; got "
            f"{tuple(target.shape)} vs {tuple(current.shape)}"
        )

    error = target - current
    processed_step = limit_vector_norm(error, max_processed_step_m)
    raw = processed_delta_to_raw_action(processed_step, action_scale)
    return raw, processed_step, error
