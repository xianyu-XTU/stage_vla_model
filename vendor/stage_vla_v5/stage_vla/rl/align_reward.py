"""Reward shaping for the modular ALIGN action policy.

The math stays independent of Isaac Lab so the safety margins can be tested
without starting the simulator.  Terminal success remains owned by the
environment; this module only supplies dense learning signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlignRewardConfig:
    """Dense ALIGN reward weights and safety margins."""

    success_xy_m: float = 0.010
    inner_xy_m: float = 0.0075
    xy_progress_weight: float = 45.0
    xy_goal_weight: float = 4.0
    xy_goal_scale_m: float = 0.025
    xy_margin_weight: float = 1.5
    height_progress_weight: float = 8.0
    height_deadband_m: float = 0.004
    height_error_weight: float = 35.0
    speed_scale_mps: float = 0.05
    speed_weight: float = 0.5
    angular_speed_scale_radps: float = 0.5
    angular_speed_weight: float = 0.35
    tilt_growth_scale_rad: float = 0.08726646259971647
    tilt_growth_weight: float = 1.0
    action_l2_weight: float = 0.12
    action_delta_weight: float = 0.08
    large_action_threshold: float = 0.65
    large_action_weight: float = 0.25
    saturation_weight: float = 0.25
    grasp_margin_start_fraction: float = 0.65
    grasp_margin_weight: float = 2.0

    def validate(self) -> None:
        positive = (
            self.success_xy_m,
            self.inner_xy_m,
            self.xy_goal_scale_m,
            self.height_deadband_m,
            self.speed_scale_mps,
            self.angular_speed_scale_radps,
            self.tilt_growth_scale_rad,
            self.large_action_threshold,
        )
        nonnegative = (
            self.xy_progress_weight,
            self.xy_goal_weight,
            self.xy_margin_weight,
            self.height_progress_weight,
            self.height_error_weight,
            self.speed_weight,
            self.angular_speed_weight,
            self.tilt_growth_weight,
            self.action_l2_weight,
            self.action_delta_weight,
            self.large_action_weight,
            self.saturation_weight,
            self.grasp_margin_weight,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("ALIGN reward distances and scales must be positive")
        if any(value < 0 for value in nonnegative):
            raise ValueError("ALIGN reward weights must be non-negative")
        if self.inner_xy_m >= self.success_xy_m:
            raise ValueError("ALIGN inner XY target must be inside the success radius")
        if self.large_action_threshold > 1.0:
            raise ValueError("ALIGN large-action threshold must not exceed one")
        if not 0.0 <= self.grasp_margin_start_fraction < 1.0:
            raise ValueError("grasp margin fraction must lie in [0,1)")


def align_reward_terms(
    before_xy_m: Tensor,
    after_xy_m: Tensor,
    before_height_error_m: Tensor,
    after_height_error_m: Tensor,
    stability_speed_mps: Tensor,
    after_angular_speed_radps: Tensor,
    before_upright_tilt_rad: Tensor,
    after_upright_tilt_rad: Tensor,
    unit_action: Tensor,
    previous_unit_action: Tensor,
    requested_action: Tensor,
    left_grasp_height_error_m: Tensor,
    right_grasp_height_error_m: Tensor,
    grasp_height_tolerance_m: Tensor,
    *,
    cfg: AlignRewardConfig,
) -> dict[str, Tensor]:
    """Return per-environment ALIGN terms before common grasp/terminal reward."""
    cfg.validate()
    after_xy = torch.as_tensor(after_xy_m)
    vectors = (
        torch.as_tensor(before_xy_m, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(before_height_error_m, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(after_height_error_m, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(stability_speed_mps, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(after_angular_speed_radps, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(before_upright_tilt_rad, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(after_upright_tilt_rad, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(left_grasp_height_error_m, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(right_grasp_height_error_m, device=after_xy.device, dtype=after_xy.dtype),
        torch.as_tensor(grasp_height_tolerance_m, device=after_xy.device, dtype=after_xy.dtype),
    )
    if after_xy.ndim != 1 or any(value.shape != after_xy.shape for value in vectors):
        raise ValueError("ALIGN reward state fields must all have shape [N]")
    unit = torch.as_tensor(unit_action, device=after_xy.device, dtype=after_xy.dtype)
    requested = torch.as_tensor(
        requested_action, device=after_xy.device, dtype=after_xy.dtype
    )
    previous = torch.as_tensor(
        previous_unit_action, device=after_xy.device, dtype=after_xy.dtype
    )
    if unit.shape != (len(after_xy), 5) or requested.shape != unit.shape or previous.shape != unit.shape:
        raise ValueError("ALIGN reward actions must all have shape [N,5]")
    if not all(torch.isfinite(value).all() for value in (after_xy, *vectors, unit, previous, requested)):
        raise ValueError("ALIGN reward inputs must be finite")
    if (
        torch.any(after_xy < 0)
        or any(torch.any(value < 0) for value in vectors[3:7])
        or torch.any(vectors[-1] <= 0)
    ):
        raise ValueError("ALIGN distances must be non-negative and tolerances positive")

    (
        before_xy,
        before_height,
        after_height,
        speed,
        angular_speed,
        before_tilt,
        after_tilt,
        left_height,
        right_height,
        tolerance,
    ) = vectors
    xy_progress = cfg.xy_progress_weight * (before_xy - after_xy).clamp(-0.05, 0.05)
    xy_goal = cfg.xy_goal_weight * torch.exp(-after_xy / cfg.xy_goal_scale_m)

    # A 10 mm boundary is too brittle for downstream DESCEND.  Keep the
    # terminal criterion unchanged, but make the outer 2.5 mm of that success
    # region carry a cost so PPO learns a usable 7.5 mm interior margin.
    margin_width = cfg.success_xy_m - cfg.inner_xy_m
    xy_margin_cost = cfg.xy_margin_weight * (
        (after_xy - cfg.inner_xy_m) / margin_width
    ).clamp(0.0, 1.0)

    height_progress = cfg.height_progress_weight * (
        before_height - after_height
    ).clamp(-0.02, 0.02)
    height_cost = cfg.height_error_weight * (
        after_height - cfg.height_deadband_m
    ).clamp_min(0.0)
    speed_cost = cfg.speed_weight * (speed / cfg.speed_scale_mps).clamp_min(0.0)
    angular_speed_cost = cfg.angular_speed_weight * (
        angular_speed / cfg.angular_speed_scale_radps
    ).clamp(0.0, 4.0)
    # Penalize making an inherited tilt worse.  Deliberately do not reward an
    # active upright correction: a side grasp can roll a heavy payload out of
    # the fingers when the wrist tries to rotate the object directly.
    tilt_growth_cost = cfg.tilt_growth_weight * (
        (after_tilt - before_tilt).clamp_min(0.0) / cfg.tilt_growth_scale_rad
    ).clamp(0.0, 4.0)

    arm_unit = unit[:, :3]
    action_cost = cfg.action_l2_weight * arm_unit.square().sum(dim=-1)
    action_delta_cost = cfg.action_delta_weight * (
        unit[:, :4] - previous[:, :4]
    ).square().sum(dim=-1)
    large_action_cost = cfg.large_action_weight * (
        unit[:, :4].abs() - cfg.large_action_threshold
    ).clamp_min(0.0).square().sum(dim=-1)
    saturation_excess = (requested[:, :3].abs() - 1.0).clamp(0.0, 4.0)
    saturation_cost = cfg.saturation_weight * saturation_excess.square().sum(dim=-1)

    max_grasp_height_error = torch.maximum(left_height, right_height)
    grasp_ratio = max_grasp_height_error / tolerance
    margin_fraction = (
        (grasp_ratio - cfg.grasp_margin_start_fraction)
        / (1.0 - cfg.grasp_margin_start_fraction)
    ).clamp(0.0, 2.0)
    grasp_margin_cost = cfg.grasp_margin_weight * margin_fraction.square()

    return {
        "xy_progress": xy_progress,
        "xy_goal": xy_goal,
        "xy_margin_cost": xy_margin_cost,
        "height_progress": height_progress,
        "height_cost": height_cost,
        "speed_cost": speed_cost,
        "angular_speed_cost": angular_speed_cost,
        "tilt_growth_cost": tilt_growth_cost,
        "action_cost": action_cost,
        "action_delta_cost": action_delta_cost,
        "large_action_cost": large_action_cost,
        "saturation_cost": saturation_cost,
        "grasp_margin_cost": grasp_margin_cost,
    }
