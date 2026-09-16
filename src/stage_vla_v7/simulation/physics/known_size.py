"""Known-size pressure control and motion metrics independent of Isaac Lab."""

from __future__ import annotations

import torch
from torch import Tensor

from stage_vla_v7.simulation.config import KnownSizeGraspConfig


def stability_speed_from_source(
    instantaneous_speed: Tensor,
    control_delta_speed: Tensor,
    source: str,
) -> Tensor:
    if instantaneous_speed.shape != control_delta_speed.shape:
        raise ValueError("instantaneous and control-delta speeds must have the same shape")
    if source == "instantaneous":
        return instantaneous_speed
    if source == "control_delta":
        return control_delta_speed
    raise ValueError("stability speed source must be instantaneous or control_delta")


def quaternion_control_angular_speed(
    before_quat_xyzw: Tensor,
    after_quat_xyzw: Tensor,
    *,
    step_dt_s: float,
) -> Tensor:
    before = torch.as_tensor(before_quat_xyzw)
    if not before.is_floating_point():
        before = before.to(dtype=torch.float32)
    after = torch.as_tensor(after_quat_xyzw, device=before.device, dtype=before.dtype)
    if before.shape != after.shape or before.ndim < 1 or before.shape[-1] != 4:
        raise ValueError("before and after quaternions must have matching [...,4] shape")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("quaternions must be finite")
    if not torch.isfinite(torch.tensor(step_dt_s)) or float(step_dt_s) <= 0.0:
        raise ValueError("step_dt_s must be positive and finite")
    before_norm = before.norm(dim=-1, keepdim=True)
    after_norm = after.norm(dim=-1, keepdim=True)
    if bool((before_norm == 0.0).any()) or bool((after_norm == 0.0).any()):
        raise ValueError("quaternions must have non-zero norm")
    before = before / before_norm
    after = after / after_norm
    same_hemisphere = torch.where(
        (before * after).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(before_norm),
        torch.ones_like(before_norm),
    )
    chord = (after - same_hemisphere * before).norm(dim=-1)
    angle = 4.0 * torch.asin((0.5 * chord).clamp(0.0, 1.0))
    return angle / float(step_dt_s)


def grasp_contact_error(
    grasp_target: Tensor,
    left_tip: Tensor,
    right_tip: Tensor,
    *,
    contact_height_m: float = 0.007,
) -> Tensor:
    target = torch.as_tensor(grasp_target)
    left = torch.as_tensor(left_tip, device=target.device, dtype=target.dtype)
    right = torch.as_tensor(right_tip, device=target.device, dtype=target.dtype)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("grasp_target must have shape [N,3]")
    if left.shape != target.shape or right.shape != target.shape:
        raise ValueError("fingertip positions must match grasp_target")
    if contact_height_m < 0 or not all(
        torch.isfinite(value).all() for value in (target, left, right)
    ):
        raise ValueError("contact height and positions must be finite and valid")
    desired = target.clone()
    desired[:, 2] += float(contact_height_m)
    return (left + right) / 2.0 - desired


def _broadcast_force(
    value: Tensor | float,
    shape: torch.Size,
    *,
    device,
    dtype,
) -> Tensor:
    force = torch.as_tensor(value, device=device, dtype=dtype)
    if force.ndim == 0:
        force = force.expand(shape)
    elif force.shape == shape[:-1]:
        force = force.unsqueeze(-1).expand(shape)
    elif force.shape != shape:
        raise ValueError(
            f"force target must be scalar, {tuple(shape[:-1])}, or {tuple(shape)}"
        )
    return force


def pressure_feedback_step(
    current_joint_pos: Tensor,
    measured_force_n: Tensor,
    target_force_n: Tensor | float,
    *,
    cfg: KnownSizeGraspConfig,
    gain_m_per_n: float = 0.0002,
    max_step_m: float = 0.001,
) -> Tensor:
    cfg.validate()
    current = torch.as_tensor(current_joint_pos)
    measured = torch.as_tensor(
        measured_force_n, device=current.device, dtype=current.dtype
    )
    if current.ndim != 2 or current.shape[-1] != 2:
        raise ValueError("current_joint_pos must have shape [N,2]")
    if measured.shape != current.shape:
        raise ValueError("measured_force_n must have shape [N,2]")
    if not torch.isfinite(current).all() or not torch.isfinite(measured).all():
        raise ValueError("joint positions and forces must be finite")
    if torch.any(measured < 0):
        raise ValueError("measured forces must be non-negative")
    if gain_m_per_n <= 0 or max_step_m <= 0:
        raise ValueError("feedback gain and max step must be positive")
    target = _broadcast_force(
        target_force_n, current.shape, device=current.device, dtype=current.dtype
    )
    if torch.any(target < cfg.min_force_n) or torch.any(target > cfg.max_force_n):
        raise ValueError("target force is outside configured safety bounds")
    error = target - measured
    delta = (-float(gain_m_per_n) * error).clamp(
        -float(max_step_m), float(max_step_m)
    )
    base = current.clamp(
        cfg.compression_joint_min_m, cfg.geometric_joint_target_m
    )
    return (base + delta).clamp(
        cfg.compression_joint_min_m, cfg.geometric_joint_target_m
    )


def pressure_target_from_grip(
    grip: Tensor,
    *,
    cfg: KnownSizeGraspConfig,
    residual_force_range_n: float | None = None,
) -> Tensor:
    cfg.validate()
    value = torch.as_tensor(grip, dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("grip command must be finite")
    residual = (
        cfg.residual_force_range_n
        if residual_force_range_n is None
        else residual_force_range_n
    )
    if residual < 0 or not torch.isfinite(torch.tensor(residual)):
        raise ValueError("residual_force_range_n must be finite and non-negative")
    closure = (-value).clamp(0.0, 1.0)
    return (cfg.initial_force_target_n + closure * float(residual)).clamp(
        cfg.min_force_n, cfg.max_force_n
    )


def size_conditioned_grasp_targets(
    cfg: KnownSizeGraspConfig,
    *,
    device=None,
    dtype=torch.float32,
) -> tuple[Tensor, Tensor]:
    cfg.validate()
    joint = torch.full(
        (2,), cfg.geometric_joint_target_m, device=device, dtype=dtype
    )
    force = torch.full((2,), cfg.initial_force_target_n, device=device, dtype=dtype)
    return joint, force


def pressure_tracking_ok(
    measured_force_n: Tensor,
    target_force_n: Tensor | float,
    *,
    cfg: KnownSizeGraspConfig,
) -> Tensor:
    cfg.validate()
    measured = torch.as_tensor(measured_force_n)
    if measured.ndim != 2 or measured.shape[-1] != 2:
        raise ValueError("measured_force_n must have shape [N,2]")
    if not torch.isfinite(measured).all() or torch.any(measured < 0):
        raise ValueError("measured_force_n must be finite and non-negative")
    target = torch.as_tensor(
        target_force_n, device=measured.device, dtype=measured.dtype
    )
    if target.ndim == 0:
        target = target.expand(measured.shape[0])
    if target.ndim == 2 and target.shape[-1] == 1:
        target = target[:, 0]
    if target.shape != (measured.shape[0],):
        raise ValueError("target_force_n must be scalar or shape [N]")
    if not torch.isfinite(target).all():
        raise ValueError("target_force_n must be finite")
    error_ok = (
        (measured - target.unsqueeze(-1)).abs().amax(dim=-1)
        <= cfg.pressure_tolerance_n
    )
    balance_ok = (
        (measured[:, 0] - measured[:, 1]).abs()
        <= cfg.force_balance_tolerance_n
    )
    return error_ok & balance_ok
