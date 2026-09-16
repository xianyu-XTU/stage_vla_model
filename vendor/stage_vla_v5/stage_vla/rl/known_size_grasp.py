"""Known-size oracle grasp geometry and pressure-control primitives.

The compact vision model is intentionally not involved here.  During the
oracle phase, object dimensions are supplied by the task configuration.  The
size produces a safe geometric jaw target and an initial per-finger force
target; measured finger forces then close the loop around that target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def stability_speed_from_source(
    instantaneous_speed: Tensor,
    control_delta_speed: Tensor,
    source: str,
) -> Tensor:
    """Select the velocity signal used by control-rate stability gates."""
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
    """Return sign-invariant angular speed over one control period."""
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
    # For unit quaternions, ||q1-q0|| = 2*sin(theta/4). This form
    # preserves small control-period rotations that acos(dot) rounds to zero.
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
    """Return fingertip-midpoint error from the GRASP contact target."""
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


@dataclass(frozen=True)
class KnownSizeGraspConfig:
    """Physical assumptions for one known-size object class."""

    width_m: float
    depth_m: float
    height_m: float
    mass_kg: float
    grasp_width_m: float | None = None
    jaw_clearance_m: float = 0.002
    max_compression_m: float = 0.004
    friction_coefficient: float = 0.4
    safety_factor: float = 2.0
    min_force_n: float = 5.0
    max_force_n: float = 40.0
    pressure_tolerance_n: float = 1.0
    force_balance_tolerance_n: float = 1.5
    residual_force_range_n: float = 4.0
    residual_action_penalty: float = 0.5
    lift_acceleration_mps2: float = 0.5
    joint_min_m: float = 0.0
    joint_max_m: float = 0.04

    def validate(self) -> None:
        grasp_width = self.width_m if self.grasp_width_m is None else self.grasp_width_m
        positive = (
            self.width_m, self.depth_m, self.height_m, self.mass_kg, grasp_width,
            self.friction_coefficient, self.safety_factor, self.min_force_n,
            self.max_force_n, self.pressure_tolerance_n,
            self.force_balance_tolerance_n, self.joint_max_m,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("size, mass, friction, force and joint limits must be positive")
        nonnegative = (self.jaw_clearance_m, self.max_compression_m,
                       self.lift_acceleration_mps2, self.joint_min_m,
                       self.residual_force_range_n, self.residual_action_penalty)
        if any(float(value) < 0 for value in nonnegative):
            raise ValueError("clearance, compression, acceleration and joint minimum must be non-negative")
        if self.min_force_n > self.max_force_n:
            raise ValueError("min_force_n must not exceed max_force_n")
        if self.joint_min_m >= self.joint_max_m:
            raise ValueError("joint_min_m must be less than joint_max_m")
        if grasp_width / 2.0 + self.jaw_clearance_m > self.joint_max_m:
            raise ValueError("object width plus clearance exceeds gripper opening")

    @property
    def geometric_joint_target_m(self) -> float:
        """Per-finger joint position at the known-size contact envelope."""
        self.validate()
        grasp_width = self.width_m if self.grasp_width_m is None else self.grasp_width_m
        return min(self.joint_max_m, grasp_width / 2.0 + self.jaw_clearance_m)

    @property
    def compression_joint_min_m(self) -> float:
        """Smallest per-finger target allowed while building pressure."""
        self.validate()
        grasp_width = self.width_m if self.grasp_width_m is None else self.grasp_width_m
        return max(self.joint_min_m, grasp_width / 2.0 - self.max_compression_m)

    @property
    def initial_force_target_n(self) -> float:
        """Conservative per-finger force prior for vertical lifting."""
        self.validate()
        required = (
            self.safety_factor * self.mass_kg
            * (9.81 + self.lift_acceleration_mps2)
            / (2.0 * self.friction_coefficient)
        )
        return min(self.max_force_n, max(self.min_force_n, required))

    def size_features(self, *, dtype=torch.float32, device=None) -> Tensor:
        """Return normalized known-size features for a size-conditioned policy."""
        self.validate()
        return torch.tensor(
            [self.width_m, self.depth_m, self.height_m], dtype=dtype, device=device
        ) / 0.1


def _broadcast_force(value: Tensor | float, shape: torch.Size, *, device, dtype) -> Tensor:
    force = torch.as_tensor(value, device=device, dtype=dtype)
    if force.ndim == 0:
        force = force.expand(shape)
    elif force.shape == shape[:-1]:
        force = force.unsqueeze(-1).expand(shape)
    elif force.shape != shape:
        raise ValueError(f"force target must be scalar, {tuple(shape[:-1])}, or {tuple(shape)}")
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
    """Update per-finger joint targets from measured contact pressure.

    Positive force error closes the fingers (smaller joint position); excess
    force opens them slightly.  The output is bounded by the known-size
    contact envelope and compression limit, so feedback cannot command an
    arbitrary squeeze.
    """
    cfg.validate()
    current = torch.as_tensor(current_joint_pos)
    measured = torch.as_tensor(measured_force_n, device=current.device, dtype=current.dtype)
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

    target = _broadcast_force(target_force_n, current.shape, device=current.device, dtype=current.dtype)
    if torch.any(target < cfg.min_force_n) or torch.any(target > cfg.max_force_n):
        raise ValueError("target force is outside configured safety bounds")
    error = target - measured
    delta = (-float(gain_m_per_n) * error).clamp(-float(max_step_m), float(max_step_m))
    # Once the fingers have reached the known-size contact envelope, keep
    # integrating pressure corrections from that envelope.  Using the raw
    # measured joint position here can pin the target at the geometric upper
    # bound when contact dynamics leave the fingers a few millimetres outside
    # the target, making pressure feedback ineffective.
    base = current.clamp(cfg.compression_joint_min_m, cfg.geometric_joint_target_m)
    next_target = base + delta
    return next_target.clamp(cfg.compression_joint_min_m, cfg.geometric_joint_target_m)


def pressure_target_from_grip(
    grip: Tensor,
    *,
    cfg: KnownSizeGraspConfig,
    residual_force_range_n: float | None = None,
) -> Tensor:
    """Map a normalized closing command to a bounded per-finger force target.

    Positive ``grip`` values still mean OPEN for the v5 interface.  For a
    closing command, ``-grip`` in ``[0, 1]`` adds a learned residual above the
    conservative size/mass prior.  The result is scalar per environment; the
    low-level controller applies it symmetrically to both fingers.
    """
    cfg.validate()
    value = torch.as_tensor(grip, dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("grip command must be finite")
    residual = cfg.residual_force_range_n if residual_force_range_n is None else residual_force_range_n
    if residual < 0 or not torch.isfinite(torch.tensor(residual)):
        raise ValueError("residual_force_range_n must be finite and non-negative")
    closure = (-value).clamp(0.0, 1.0)
    return (cfg.initial_force_target_n + closure * float(residual)).clamp(
        cfg.min_force_n, cfg.max_force_n
    )


def known_size_raw_action(
    policy_action: Tensor,
    *,
    translation_limit_m: float = 0.005,
    yaw_limit_rad: float = 0.02,
) -> tuple[Tensor, Tensor]:
    """Decode v5 ``[dx,dy,dz,dyaw,grip]`` while preserving grip magnitude."""
    if policy_action.ndim != 2 or policy_action.shape[-1] != 5:
        raise ValueError("known-size action must have shape [N,5]")
    if translation_limit_m <= 0 or yaw_limit_rad <= 0:
        raise ValueError("action limits must be positive")
    if not torch.isfinite(policy_action).all():
        raise ValueError("known-size action contains NaN/Inf")
    unit = policy_action.clamp(-1.0, 1.0)
    raw = torch.zeros((unit.shape[0], 7), device=unit.device, dtype=unit.dtype)
    raw[:, :3] = unit[:, :3] * float(translation_limit_m)
    raw[:, 5] = unit[:, 3] * float(yaw_limit_rad)
    # Unlike direct_action(), preserve the negative magnitude for pressure
    # control. The custom action term still interprets positive as OPEN.
    raw[:, 6] = unit[:, 4]
    return raw, unit


def size_conditioned_grasp_targets(
    cfg: KnownSizeGraspConfig, *, device=None, dtype=torch.float32
) -> tuple[Tensor, Tensor]:
    """Return initial ``([left_q,right_q], [left_N,right_N])`` targets."""
    cfg.validate()
    joint = torch.full((2,), cfg.geometric_joint_target_m, device=device, dtype=dtype)
    force = torch.full((2,), cfg.initial_force_target_n, device=device, dtype=dtype)
    return joint, force


def pressure_tracking_ok(
    measured_force_n: Tensor,
    target_force_n: Tensor | float,
    *,
    cfg: KnownSizeGraspConfig,
) -> Tensor:
    """Return a per-environment pressure-and-balance acceptance mask."""
    cfg.validate()
    measured = torch.as_tensor(measured_force_n)
    if measured.ndim != 2 or measured.shape[-1] != 2:
        raise ValueError("measured_force_n must have shape [N,2]")
    if not torch.isfinite(measured).all() or torch.any(measured < 0):
        raise ValueError("measured_force_n must be finite and non-negative")
    target = torch.as_tensor(target_force_n, device=measured.device, dtype=measured.dtype)
    if target.ndim == 0:
        target = target.expand(measured.shape[0])
    if target.ndim == 2 and target.shape[-1] == 1:
        target = target[:, 0]
    if target.shape != (measured.shape[0],):
        raise ValueError("target_force_n must be scalar or shape [N]")
    if not torch.isfinite(target).all():
        raise ValueError("target_force_n must be finite")
    error_ok = (measured - target.unsqueeze(-1)).abs().amax(dim=-1) <= cfg.pressure_tolerance_n
    balance_ok = (measured[:, 0] - measured[:, 1]).abs() <= cfg.force_balance_tolerance_n
    return error_ok & balance_ok
