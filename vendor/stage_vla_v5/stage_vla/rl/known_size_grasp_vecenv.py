"""Standalone oracle GRASP/LIFT environment with learned pressure residuals.

This wrapper is intentionally narrow: object dimensions are fixed by
``KnownSizeGraspConfig`` and the policy is evaluated without any visual input.
The first experiment can lock the first four arm action channels and train only
the continuous gripper-pressure residual while retaining the v5 five-value
action interface.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from stage_vla.envs.state_readers import read_grasp_state, read_placement_state, to_torch
from stage_vla.rl.known_size_grasp import (
    KnownSizeGraspConfig,
    grasp_contact_error,
    known_size_raw_action,
    quaternion_control_angular_speed,
    stability_speed_from_source,
)
from stage_vla.rl.align_reward import AlignRewardConfig, align_reward_terms
from stage_vla.rl.object_physics import (
    PhysicalObjectBatch,
    snapshot_has_physical_context,
)
from stage_vla.rl.skill_action_safety import (
    carrying_grasp_lost,
    entrance_action_scale,
    jaw_leveling_axis_angle,
    object_upright_tilt_rad,
    project_align_motion,
    project_descend_motion,
    project_pregrasp_edge_alignment,
)
from stage_vla.rl.transport_handoff import (
    TransportHandoffConfig,
    transport_handoff_ready,
    transport_settle_reward_terms,
)
from stage_vla.stages.physical_grasp import PhysicalGraspConfig, physical_grasp_diagnostics
from stage_vla.stages.grasp_geometry import (
    GraspGeometryProfile,
    grasp_target_position,
    parallel_jaw_yaw_error,
)


LEGACY_OBS_DIM = 55
PHYSICAL_CONTEXT_DIM = 10
CONTEXT_OBS_DIM = LEGACY_OBS_DIM + PHYSICAL_CONTEXT_DIM
# Kept for callers that load existing 55-D policies.
OBS_DIM = LEGACY_OBS_DIM
ENTRANCE_RECOVERY_MAX_DAMPING_ACCELERATION_MPS2 = 1.0
GRASP_TOUCH_FORCE_THRESHOLD_N = 0.05


def entrance_contact_ready(
    between_fingertips: torch.Tensor,
    finger_a_contact: torch.Tensor,
    finger_b_contact: torch.Tensor,
    pressure_ok: torch.Tensor,
) -> torch.Tensor:
    """Return the physical condition that completes downstream entrance recovery."""
    return between_fingertips & finger_a_contact & finger_b_contact & pressure_ok


def update_entrance_contact_recovered(
    recovered: torch.Tensor,
    contact_ready: torch.Tensor,
) -> torch.Tensor:
    """Latch contact recovery; losing contact never clears the entrance latch."""
    return recovered | contact_ready


def update_grasp_contact_seen(
    seen: torch.Tensor,
    fingertip_force_n: torch.Tensor,
    active: torch.Tensor,
    *,
    threshold_n: float = GRASP_TOUCH_FORCE_THRESHOLD_N,
) -> torch.Tensor:
    """Latch the first light fingertip touch used only to freeze GRASP wrist motion."""
    seen = torch.as_tensor(seen, dtype=torch.bool)
    force = torch.as_tensor(fingertip_force_n, device=seen.device)
    active = torch.as_tensor(active, device=seen.device, dtype=torch.bool)
    if force.ndim != seen.ndim + 1 or force.shape[:-1] != seen.shape:
        raise ValueError("fingertip_force_n must add one fingertip axis to seen")
    if force.shape[-1] < 1:
        raise ValueError("fingertip_force_n must contain at least one fingertip")
    if active.shape != seen.shape:
        raise ValueError("active and seen must have the same shape")
    if not np.isfinite(threshold_n) or float(threshold_n) <= 0.0:
        raise ValueError("threshold_n must be positive and finite")
    light_touch = force.max(dim=-1).values > float(threshold_n)
    return seen | (active & light_touch)


def entrance_recovery_active(
    recovered: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return environments still covered by one-shot entrance recovery."""
    recovered = torch.as_tensor(recovered, dtype=torch.bool)
    active = torch.as_tensor(active, device=recovered.device, dtype=torch.bool)
    if recovered.shape != active.shape:
        raise ValueError("recovered and active must have the same shape")
    return ~recovered & active


def entrance_grasp_recovery_ready(
    contact_ready: torch.Tensor,
    left_height_aligned: torch.Tensor,
    right_height_aligned: torch.Tensor,
    height_recovery_complete: torch.Tensor,
) -> torch.Tensor:
    """Require original contact gates and the existing physical height gates."""
    contact_ready = torch.as_tensor(contact_ready, dtype=torch.bool)
    left = torch.as_tensor(
        left_height_aligned, device=contact_ready.device, dtype=torch.bool
    )
    right = torch.as_tensor(
        right_height_aligned, device=contact_ready.device, dtype=torch.bool
    )
    centered = torch.as_tensor(
        height_recovery_complete, device=contact_ready.device, dtype=torch.bool
    )
    if any(
        value.shape != contact_ready.shape for value in (left, right, centered)
    ):
        raise ValueError("contact and height readiness must have the same shape")
    return contact_ready & left & right & centered


def update_entrance_height_recovery(
    required: torch.Tensor,
    complete: torch.Tensor,
    height_aligned: torch.Tensor,
    height_centered: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latch which entrances need regrasp and when their correction completes."""
    required = torch.as_tensor(required, dtype=torch.bool)
    values = [
        torch.as_tensor(value, device=required.device, dtype=torch.bool)
        for value in (complete, height_aligned, height_centered, eligible)
    ]
    if any(value.shape != required.shape for value in values):
        raise ValueError("entrance height-recovery masks must have the same shape")
    complete, height_aligned, height_centered, eligible = values
    required = required | (eligible & ~height_aligned & ~complete)
    complete = complete | (
        eligible
        & height_aligned
        & ((~required) | height_centered)
    )
    return required, complete


def entrance_fingertip_height_step(
    grasp_target: torch.Tensor,
    left_tip: torch.Tensor,
    right_tip: torch.Tensor,
    recovered: torch.Tensor,
    *,
    max_step_m: float,
) -> torch.Tensor:
    """Return a bounded wrist-Z correction toward the live grasp height."""
    target = torch.as_tensor(grasp_target)
    left = torch.as_tensor(left_tip, device=target.device, dtype=target.dtype)
    right = torch.as_tensor(right_tip, device=target.device, dtype=target.dtype)
    recovered = torch.as_tensor(recovered, device=target.device, dtype=torch.bool)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("grasp_target must have shape [N,3]")
    if left.shape != target.shape or right.shape != target.shape:
        raise ValueError("fingertip positions must match grasp_target")
    if recovered.shape != (len(target),):
        raise ValueError("recovered must have shape [N]")
    if (
        not np.isfinite(max_step_m)
        or float(max_step_m) <= 0.0
        or not all(torch.isfinite(value).all() for value in (target, left, right))
    ):
        raise ValueError("positions and max_step_m must be finite; max_step_m must be positive")
    midpoint_z = (left[:, 2] + right[:, 2]) / 2.0
    correction = (target[:, 2] - midpoint_z).clamp(
        min=-float(max_step_m), max=float(max_step_m)
    )
    return torch.where(recovered, torch.zeros_like(correction), correction)


def entrance_fingertip_height_ready(
    grasp_target: torch.Tensor,
    left_tip: torch.Tensor,
    right_tip: torch.Tensor,
    *,
    tolerance_m: float,
) -> torch.Tensor:
    """Return whether the fingertip midpoint is centered for entrance regrasp."""
    if not np.isfinite(tolerance_m) or float(tolerance_m) <= 0.0:
        raise ValueError("tolerance_m must be positive and finite")
    target = torch.as_tensor(grasp_target)
    left = torch.as_tensor(left_tip, device=target.device, dtype=target.dtype)
    right = torch.as_tensor(right_tip, device=target.device, dtype=target.dtype)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("grasp_target must have shape [N,3]")
    if left.shape != target.shape or right.shape != target.shape:
        raise ValueError("fingertip positions must match grasp_target")
    if not all(torch.isfinite(value).all() for value in (target, left, right)):
        raise ValueError("fingertip positions must be finite")
    midpoint_error = ((left[:, 2] + right[:, 2]) / 2.0 - target[:, 2]).abs()
    return midpoint_error <= float(tolerance_m)


def entrance_velocity_damping_force(
    object_mass_kg: torch.Tensor,
    linear_velocity_mps: torch.Tensor,
    recovery_active: torch.Tensor,
    *,
    step_dt_s: float,
    max_acceleration_mps2: float,
) -> torch.Tensor:
    """Return a bounded entrance-only force opposing restored payload drift."""
    mass = torch.as_tensor(object_mass_kg)
    velocity = torch.as_tensor(
        linear_velocity_mps, device=mass.device, dtype=mass.dtype
    )
    active = torch.as_tensor(recovery_active, device=mass.device, dtype=torch.bool)
    if mass.ndim != 1 or velocity.shape != (len(mass), 3) or active.shape != mass.shape:
        raise ValueError("mass, velocity and recovery mask shapes are inconsistent")
    if (
        not torch.isfinite(mass).all()
        or not torch.isfinite(velocity).all()
        or torch.any(mass <= 0.0)
        or not np.isfinite(step_dt_s)
        or float(step_dt_s) <= 0.0
        or not np.isfinite(max_acceleration_mps2)
        or float(max_acceleration_mps2) <= 0.0
    ):
        raise ValueError("damping inputs must be finite and strictly positive")
    acceleration = -velocity / float(step_dt_s)
    norm = acceleration.norm(dim=-1, keepdim=True)
    scale = (float(max_acceleration_mps2) / norm.clamp_min(1.0e-9)).clamp_max(1.0)
    force = mass[:, None] * acceleration * scale
    return torch.where(active[:, None], force, torch.zeros_like(force))


class KnownSizeGraspVecEnv(RslRlVecEnvWrapper):
    """Vectorized known-size physical-skill force-control task."""

    def __init__(
        self,
        env,
        *,
        known_size: KnownSizeGraspConfig,
        skill: str = "GRASP",
        snapshots=(),
        episode_steps: int = 80,
        stable_steps: int = 3,
        lift_height_m: float = 0.06,
        lift_translation_limit_m: float = 0.005,
        arm_locked: bool = True,
        lift_vertical_only: bool = False,
        lift_prepress_steps: int = 0,
        lift_prepress_grip: float = -1.0,
        entrance_prepress_steps: int = 0,
        entrance_prepress_grip: float = -1.0,
        entrance_gravity_compensation: bool = True,
        entrance_gravity_compensation_until_contact: bool = True,
        entrance_contact_refresh_steps: int = 0,
        entrance_servo_target_blend: float = 1.0,
        entrance_servo_target_max_delta_m: float | None = None,
        entrance_arm_warmup_steps: int = 0,
        entrance_arm_warmup_scale: float = 1.0,
        lift_arm_warmup_steps: int = 0,
        lift_arm_warmup_scale: float = 0.2,
        entrance_height_correction_max_step_m: float = 0.005,
        entrance_height_correction_tolerance_m: float = 0.002,
        carry_leveling_max_angle_rad: float = 0.0,
        pregrasp_yaw_tolerance_rad: float = 0.005,
        transport_planar_only: bool = True,
        transport_xy_m: float = 0.045,
        transport_speed_mps: float = 0.05,
        transport_angular_speed_radps: float = 1.0,
        transport_reward_variant: str = "progress_direction_v2",
        transport_translation_limit_m: float = 0.005,
        align_xy_m: float = 0.010,
        align_inner_xy_m: float = 0.0075,
        align_height_target_m: float = 0.0618,
        align_height_tolerance_m: float = 0.015,
        align_speed_mps: float = 0.05,
        align_angular_speed_radps: float = 0.2,
        align_upright_tolerance_rad: float = 0.08726646259971647,
        align_translation_limit_m: float = 0.005,
        descend_height_target_m: float | None = None,
        descend_height_tolerance_m: float = 0.008,
        descend_speed_mps: float = 0.05,
        descend_translation_limit_m: float = 0.003,
        release_xy_m: float = 0.040,
        release_height_tolerance_m: float = 0.010,
        release_speed_mps: float = 0.05,
        release_translation_limit_m: float = 0.003,
        retreat_distance_m: float = 0.100,
        retreat_height_m: float = 0.080,
        retreat_speed_mps: float = 0.05,
        retreat_translation_limit_m: float = 0.005,
        stability_speed_source: str = "instantaneous",
        seed: int = 17,
        snapshot_assignment: str = "random",
        grasp_profile: GraspGeometryProfile | None = None,
        object_asset_name: str = "cube_2",
        support_asset_name: str = "cube_1",
        physical_batch: PhysicalObjectBatch | None = None,
        include_physical_context: bool = False,
    ):
        if skill not in ("GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND",
                         "RELEASE_STABILIZE", "RETREAT"):
            raise ValueError("unsupported known-size skill")
        if episode_steps < 1 or stable_steps < 1:
            raise ValueError("episode_steps and stable_steps must be positive")
        if lift_prepress_steps < 0:
            raise ValueError("lift_prepress_steps must be non-negative")
        if not np.isfinite(lift_prepress_grip) or not -1.0 <= lift_prepress_grip <= 1.0:
            raise ValueError("lift_prepress_grip must be finite and in [-1,1]")
        if entrance_prepress_steps < 0:
            raise ValueError("entrance_prepress_steps must be non-negative")
        if entrance_contact_refresh_steps < 0:
            raise ValueError("entrance_contact_refresh_steps must be non-negative")
        if entrance_arm_warmup_steps < 0:
            raise ValueError("entrance_arm_warmup_steps must be non-negative")
        if not 0.0 < float(entrance_arm_warmup_scale) <= 1.0:
            raise ValueError("entrance_arm_warmup_scale must lie in (0,1]")
        if lift_arm_warmup_steps < 0:
            raise ValueError("lift_arm_warmup_steps must be non-negative")
        if not 0.0 < float(lift_arm_warmup_scale) <= 1.0:
            raise ValueError("lift_arm_warmup_scale must lie in (0,1]")
        if (
            not np.isfinite(entrance_height_correction_max_step_m)
            or float(entrance_height_correction_max_step_m) <= 0.0
        ):
            raise ValueError(
                "entrance_height_correction_max_step_m must be positive and finite"
            )
        if (
            not np.isfinite(entrance_height_correction_tolerance_m)
            or float(entrance_height_correction_tolerance_m) <= 0.0
        ):
            raise ValueError(
                "entrance_height_correction_tolerance_m must be positive and finite"
            )
        if (
            not np.isfinite(carry_leveling_max_angle_rad)
            or not 0.0 <= float(carry_leveling_max_angle_rad) <= np.pi / 2.0
        ):
            raise ValueError("carry_leveling_max_angle_rad must lie in [0,pi/2]")
        if (
            not np.isfinite(pregrasp_yaw_tolerance_rad)
            or not 0.0 <= float(pregrasp_yaw_tolerance_rad) < np.pi / 4.0
        ):
            raise ValueError("pregrasp_yaw_tolerance_rad must lie in [0,pi/4)")
        if not 0.0 <= float(entrance_servo_target_blend) <= 1.0:
            raise ValueError("entrance_servo_target_blend must lie in [0,1]")
        if entrance_servo_target_max_delta_m is not None and (
            not np.isfinite(entrance_servo_target_max_delta_m)
            or float(entrance_servo_target_max_delta_m) <= 0.0
        ):
            raise ValueError(
                "entrance_servo_target_max_delta_m must be positive and finite"
            )
        if (
            not np.isfinite(entrance_prepress_grip)
            or not -1.0 <= entrance_prepress_grip <= 1.0
        ):
            raise ValueError("entrance_prepress_grip must be finite and in [-1,1]")
        if min(transport_xy_m, transport_speed_mps, transport_angular_speed_radps) <= 0:
            raise ValueError("TRANSPORT handoff thresholds must be positive")
        if transport_reward_variant not in (
            "absolute_v1", "progress_direction_v2", "progress_safety_v3",
            "progress_height_hold_v4", "progress_settle_v5",
        ):
            raise ValueError(
                "transport_reward_variant must be absolute_v1, progress_direction_v2, "
                "progress_safety_v3, progress_height_hold_v4 or progress_settle_v5"
            )
        if transport_translation_limit_m <= 0:
            raise ValueError("transport_translation_limit_m must be positive")
        if min(
            align_xy_m,
            align_inner_xy_m,
            align_height_tolerance_m,
            align_speed_mps,
            align_angular_speed_radps,
            align_upright_tolerance_rad,
        ) <= 0:
            raise ValueError("ALIGN tolerances must be positive")
        if align_upright_tolerance_rad > np.pi / 2.0:
            raise ValueError("ALIGN upright tolerance must not exceed pi/2")
        if align_inner_xy_m >= align_xy_m:
            raise ValueError("ALIGN inner XY target must be inside the success radius")
        if align_translation_limit_m <= 0:
            raise ValueError("ALIGN translation limit must be positive")
        if min(descend_height_tolerance_m, descend_speed_mps, descend_translation_limit_m,
               release_xy_m, release_height_tolerance_m, release_speed_mps,
               release_translation_limit_m, retreat_distance_m, retreat_height_m,
               retreat_speed_mps, retreat_translation_limit_m) <= 0:
            raise ValueError("skill tolerances, limits and retreat clearance must be positive")
        if stability_speed_source not in ("instantaneous", "control_delta"):
            raise ValueError(
                "stability_speed_source must be instantaneous or control_delta"
            )
        if min(lift_height_m, lift_translation_limit_m) <= 0:
            raise ValueError("lift height and translation limit must be positive")
        known_size.validate()
        super().__init__(env, clip_actions=1.0)
        self.num_actions = 5
        self.max_episode_length = int(episode_steps)
        term = self.unwrapped.action_manager.get_term("gripper_action")
        from stage_vla.envs.known_size_grasp_action import KnownSizeGraspAction
        if not isinstance(term, KnownSizeGraspAction):
            raise ValueError("environment must use KnownSizeGraspAction with the same oracle size")
        self.gripper_action = term
        self.known_size = known_size
        if physical_batch is not None:
            if not isinstance(physical_batch, PhysicalObjectBatch):
                raise TypeError("physical_batch must be a PhysicalObjectBatch")
            if physical_batch.num_envs != self.num_envs:
                raise ValueError("physical_batch size must equal num_envs")
            expected = physical_batch.to(self.device).action_context()
            actual = term.physical_batch.action_context()
            if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
                raise ValueError("controller and wrapper physical profiles do not match")
        self.physical_batch = term.physical_batch
        self.geometry_adapter = term.geometry_adapter
        self.load_adapter = term.load_adapter
        self.include_physical_context = bool(include_physical_context)
        if self.include_physical_context and physical_batch is None:
            raise ValueError("context-aware observations require an explicit physical_batch")
        self.observation_dim = (
            CONTEXT_OBS_DIM if self.include_physical_context else LEGACY_OBS_DIM
        )
        self.grasp_profile = grasp_profile or GraspGeometryProfile()
        self.grasp_profile.validate()
        self.skill = skill
        self.episode_steps = int(episode_steps)
        self.stable_steps = int(stable_steps)
        self.lift_height_m = float(lift_height_m)
        self.lift_translation_limit_m = float(lift_translation_limit_m)
        self.arm_locked = bool(arm_locked)
        self.lift_vertical_only = bool(lift_vertical_only)
        self.lift_prepress_steps = int(lift_prepress_steps)
        self.lift_prepress_grip = float(lift_prepress_grip)
        self.entrance_prepress_steps = int(entrance_prepress_steps)
        self.entrance_prepress_grip = float(entrance_prepress_grip)
        self.entrance_gravity_compensation = bool(entrance_gravity_compensation)
        self.entrance_gravity_compensation_until_contact = bool(
            entrance_gravity_compensation_until_contact
        )
        self.entrance_contact_refresh_steps = min(
            int(entrance_contact_refresh_steps), self.entrance_prepress_steps
        )
        self.entrance_servo_target_blend = float(entrance_servo_target_blend)
        self.entrance_servo_target_max_delta_m = float(
            known_size.max_compression_m
            if entrance_servo_target_max_delta_m is None
            else entrance_servo_target_max_delta_m
        )
        self.entrance_arm_warmup_steps = int(entrance_arm_warmup_steps)
        self.entrance_arm_warmup_scale = float(entrance_arm_warmup_scale)
        self.lift_arm_warmup_steps = int(lift_arm_warmup_steps)
        self.lift_arm_warmup_scale = float(lift_arm_warmup_scale)
        self.entrance_height_correction_max_step_m = float(
            entrance_height_correction_max_step_m
        )
        self.entrance_height_correction_tolerance_m = float(
            entrance_height_correction_tolerance_m
        )
        self.carry_leveling_max_angle_rad = float(carry_leveling_max_angle_rad)
        self.pregrasp_yaw_tolerance_rad = float(pregrasp_yaw_tolerance_rad)
        if self.skill == "LIFT":
            self.prepress_steps = self.lift_prepress_steps
            self.prepress_grip = self.lift_prepress_grip
        elif self.skill in ("TRANSPORT", "ALIGN", "DESCEND", "RELEASE_STABILIZE"):
            self.prepress_steps = self.entrance_prepress_steps
            self.prepress_grip = self.entrance_prepress_grip
        else:
            self.prepress_steps = 0
            self.prepress_grip = self.entrance_prepress_grip
        self.transport_planar_only = bool(transport_planar_only)
        self.transport_xy_m = float(transport_xy_m)
        self.transport_speed_mps = float(transport_speed_mps)
        self.transport_angular_speed_radps = float(transport_angular_speed_radps)
        self.transport_reward_variant = transport_reward_variant
        self.transport_translation_limit_m = float(transport_translation_limit_m)
        self.transport_handoff_cfg = TransportHandoffConfig(
            success_xy_m=self.transport_xy_m,
            linear_speed_mps=self.transport_speed_mps,
            angular_speed_radps=self.transport_angular_speed_radps,
            approach_width_m=self.transport_xy_m,
        )
        self.transport_handoff_cfg.validate()
        self.align_xy_m = float(align_xy_m)
        self.align_inner_xy_m = float(align_inner_xy_m)
        self.align_height_target_m = float(align_height_target_m)
        self.align_height_tolerance_m = float(align_height_tolerance_m)
        self.align_speed_mps = float(align_speed_mps)
        self.align_angular_speed_radps = float(align_angular_speed_radps)
        self.align_upright_tolerance_rad = float(align_upright_tolerance_rad)
        self.align_translation_limit_m = float(align_translation_limit_m)
        self.align_reward_cfg = AlignRewardConfig(
            success_xy_m=self.align_xy_m,
            inner_xy_m=self.align_inner_xy_m,
            speed_scale_mps=self.align_speed_mps,
        )
        self.align_reward_cfg.validate()
        if physical_batch is not None and descend_height_target_m is None:
            self.descend_target_mode = "per_environment_stack_center_separation"
        else:
            self.descend_target_mode = "fixed"
        self.descend_height_target_m = (
            None if self.descend_target_mode != "fixed"
            else float(
                known_size.height_m
                if descend_height_target_m is None else descend_height_target_m
            )
        )
        if physical_batch is not None and descend_height_target_m is None:
            self.descend_target_height_m = self.geometry_adapter.stack_center_separation_m.clone()
        else:
            self.descend_target_height_m = torch.full(
                (self.num_envs,), float(self.descend_height_target_m), device=self.device
            )
        nominal_stack_separation = (
            float(known_size.height_m) + 0.04
        ) / 2.0
        align_clearance = self.align_height_target_m - nominal_stack_separation
        self.align_target_height_m = (
            self.geometry_adapter.stack_center_separation_m + align_clearance
            if physical_batch is not None
            else torch.full(
                (self.num_envs,), self.align_height_target_m, device=self.device
            )
        )
        self.descend_height_tolerance_m = float(descend_height_tolerance_m)
        self.descend_speed_mps = float(descend_speed_mps)
        self.descend_translation_limit_m = float(descend_translation_limit_m)
        self.release_xy_m = float(release_xy_m)
        self.release_height_tolerance_m = float(release_height_tolerance_m)
        self.release_speed_mps = float(release_speed_mps)
        self.release_translation_limit_m = float(release_translation_limit_m)
        self.retreat_distance_m = float(retreat_distance_m)
        self.retreat_height_m = float(retreat_height_m)
        self.retreat_speed_mps = float(retreat_speed_mps)
        self.retreat_translation_limit_m = float(retreat_translation_limit_m)
        self.stability_speed_source = stability_speed_source
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.snapshot_paths = [str(Path(path).resolve()) for path in snapshots]
        self.payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in snapshots]
        if snapshot_assignment not in ("random", "sequential"):
            raise ValueError("snapshot_assignment must be random or sequential")
        self.snapshot_assignment = snapshot_assignment
        for payload in self.payloads:
            if "scene_state" not in payload:
                raise ValueError("known-size entries must contain scene_state")
        context_flags = [snapshot_has_physical_context(payload) for payload in self.payloads]
        if self.include_physical_context and any(context_flags):
            if not all(context_flags):
                raise ValueError("context-aware snapshot batches cannot mix legacy and physical entries")
            if self.snapshot_assignment != "sequential":
                raise ValueError(
                    "physical snapshots must use sequential assignment so pose and physics stay bound"
                )
            snapshot_batch = PhysicalObjectBatch.from_snapshot_payloads(
                self.payloads, self.num_envs
            )
            if snapshot_batch.geometry_bundle != self.physical_batch.geometry_bundle:
                raise ValueError("snapshot and environment geometry bundles do not match")
            if not torch.allclose(
                snapshot_batch.action_context(),
                self.physical_batch.to("cpu").action_context(),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("snapshot physical contexts do not match environment profiles")
        self.scene = self.unwrapped.scene
        self.robot = self.scene["robot"]
        if object_asset_name == support_asset_name:
            raise ValueError("object and support assets must differ")
        self.object_asset_name = str(object_asset_name)
        self.support_asset_name = str(support_asset_name)
        self.red = self.scene[self.object_asset_name]
        self.blue = self.scene[self.support_asset_name]
        self.arm_ids, _ = self.robot.find_joints(["panda_joint[1-7]"], preserve_order=True)
        if len(self.arm_ids) != 7:
            raise ValueError("expected seven Franka arm joints")
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stable_count = torch.zeros_like(self.steps)
        self.prev_force = torch.zeros(self.num_envs, 2, device=self.device)
        self.entry_red_z = torch.zeros(self.num_envs, device=self.device)
        self.lift_target_height_m = torch.full(
            (self.num_envs,), self.lift_height_m, device=self.device
        )
        self.prev_unit = torch.zeros(self.num_envs, 5, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Downstream snapshots need a short, deterministic grasp-rebuild
        # phase. This latch makes that phase one-shot per episode: after real
        # contact and valid grasp height have both been observed, later drops
        # must reach normal failure.
        self.entrance_contact_recovered = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.grasp_contact_seen = torch.zeros_like(
            self.entrance_contact_recovered
        )
        # This lower threshold only freezes wrist motion after first touch.
        # Physical grasp success still requires the independent 0.5 N gate.
        self.grasp_touch_force_threshold_n = GRASP_TOUCH_FORCE_THRESHOLD_N
        # Snapshot restoration loses PhysX contact caches and therefore needs
        # one-shot entrance conditioning. A continuous skill handoff keeps
        # those caches and must not replay the restoration-only conditioner.
        self.snapshot_recovery_active = True
        self.entrance_height_recovery_required = torch.zeros_like(
            self.entrance_contact_recovered
        )
        self.entrance_height_recovery_complete = torch.zeros_like(
            self.entrance_contact_recovered
        )
        self.last_success = torch.zeros_like(self.finished)
        self.last_failure = torch.zeros_like(self.finished)
        self.last_timeout = torch.zeros_like(self.finished)
        self.auto_reset = True
        self.measured = None
        # Optional visual object poses used only in policy observations. The
        # measured dictionary remains simulator truth for rewards and terminal
        # predicates, so this adapter cannot silently turn vision into oracle
        # physical feedback.
        self.visual_red = None
        self.visual_blue = None
        self.total_steps = self.completed = self.successes = self.failures = self.timeouts = 0
        # The side contact point of a tapered object is reconstructed from a
        # local geometry profile.  Its simulator fingertip frame has a small
        # fixed vertical offset, so use a profile-specific tolerance instead
        # of weakening the box/cylinder contract globally.
        height_tolerance_m = 0.015 if self.grasp_profile.geometry == "cone" else 0.012
        self.physical_cfg = PhysicalGraspConfig(
            radial_tolerance_m=0.03,
            height_tolerance_m=height_tolerance_m,
            contact_force_threshold_n=0.5,
            endpoint_margin=0.05,
        )
        self.physical_height_tolerance_m = (
            (
                self.physical_batch.object_size_m[:, 2] / 2.0 - 0.002
            ).clamp_min(self.physical_cfg.height_tolerance_m)
            if self.grasp_profile.geometry == "box"
            else torch.full(
                (self.num_envs,), self.physical_cfg.height_tolerance_m,
                device=self.device,
            )
        )
        self.last_align_reward_terms: dict[str, torch.Tensor] = {}
        self.last_transport_reward_terms: dict[str, torch.Tensor] = {}
        # The wrapper owns a compact policy observation. Keep Isaac's
        # native 7-D action space intact so native REACH stepping remains
        # stable when a camera is active.
        self.unwrapped.single_observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Box(
                -10.0, 10.0, (self.observation_dim,), dtype=np.float32
            )
        })
        self.unwrapped.observation_space = gym.vector.utils.batch_space(
            self.unwrapped.single_observation_space, self.num_envs
        )

    def _measure(self):
        placement = read_placement_state(
            self.unwrapped,
            red_cube_name=self.object_asset_name,
            blue_cube_name=self.support_asset_name,
        )
        grasp = read_grasp_state(
            self.unwrapped, red_cube_name=self.object_asset_name
        )
        force = torch.stack([grasp.finger_a_force_n, grasp.finger_b_force_n], dim=-1)
        grasp_target = grasp_target_position(
            grasp.red_pos_w,
            self.physical_batch.object_size_m,
            profile=self.grasp_profile,
        )
        physical_diag = physical_grasp_diagnostics(
            grasp.red_pos_w,
            grasp.left_tip_w,
            grasp.right_tip_w,
            grasp.finger_a_force_n,
            grasp.finger_b_force_n,
            cfg=self.physical_cfg,
            reference_pos_w=grasp_target,
            # Keep both fingertip frames at least 2 mm inside a box's top and
            # bottom faces.  This same tolerance feeds the ALIGN safety-margin
            # reward, so reward shaping and the physical terminal agree.
            height_tolerance_m=self.physical_height_tolerance_m,
        )
        red_quat = to_torch(self.red.data.root_quat_w).clone()
        blue_quat = to_torch(self.blue.data.root_quat_w).clone()
        return {
            "red": placement.red_pos_w.clone(),
            "grasp_target": grasp_target,
            "blue": placement.blue_pos_w.clone(),
            "vel": placement.red_lin_vel_w.clone(),
            "angular": placement.red_ang_vel_w.clone(),
            "red_quat": red_quat,
            "blue_quat": blue_quat,
            "open": placement.gripper_open.clone(),
            "grip": placement.gripper_joint_pos.clone(),
            "speed": placement.red_lin_vel_w.norm(dim=-1),
            "ee": grasp.ee_pos_w.clone(),
            "left_tip": grasp.left_tip_w.clone(),
            "right_tip": grasp.right_tip_w.clone(),
            "force": force,
            "physical": physical_diag.physical_grasp,
            "between_fingertips": physical_diag.between_fingertips,
            "left_height_aligned": physical_diag.left_height_aligned,
            "right_height_aligned": physical_diag.right_height_aligned,
            "finger_a_contact": physical_diag.finger_a_contact,
            "finger_b_contact": physical_diag.finger_b_contact,
            "radial_error_m": physical_diag.radial_error_m,
            "left_height_error_m": physical_diag.left_height_error_m,
            "right_height_error_m": physical_diag.right_height_error_m,
            "projection_alpha": physical_diag.projection_alpha,
            "fingertip_gap_m": physical_diag.fingertip_gap_m,
            "q": to_torch(self.robot.data.joint_pos)[:, self.arm_ids].clone(),
            "qd": to_torch(self.robot.data.joint_vel)[:, self.arm_ids].clone(),
        }

    def _restore(self, ids):
        if not len(ids):
            return
        legacy_grasp_ids = []
        if self.payloads:
            if self.snapshot_assignment == "sequential":
                # Keep per-environment physical profiles bound to the snapshot
                # captured for that environment, including partial auto-reset
                # batches.  Restarting at payload zero for every subset mixes
                # object size/mass with an unrelated restored gripper pose.
                choices = ids.detach().to(device="cpu", dtype=torch.long) % len(
                    self.payloads
                )
            else:
                choices = torch.randint(len(self.payloads), (len(ids),), generator=self.rng)
            for choice in choices.unique().tolist():
                mask = choices == choice
                selected = ids[mask.to(self.device)]
                from stage_vla.rl.place_snapshot import expand_single_env_state

                payload = self.payloads[choice]
                state = expand_single_env_state(
                    payload["scene_state"], len(selected), self.device
                )
                self.unwrapped.reset_to(state, env_ids=selected, is_relative=True)
                if self.skill != "GRASP":
                    restored_joints = state["articulation"]["robot"][
                        "joint_position"
                    ]
                    servo_target = payload.get("controller_state", {}).get(
                        "gripper_joint_target_m"
                    )
                    if servo_target is not None:
                        servo_target = torch.as_tensor(
                            servo_target,
                            device=self.device,
                            dtype=torch.float32,
                        ).reshape(1, 2).expand(len(selected), 2)
                    self.gripper_action.prime_closed(
                        selected,
                        joint_position=restored_joints,
                        servo_target=servo_target,
                        target_blend=self.entrance_servo_target_blend,
                        max_target_delta_m=self.entrance_servo_target_max_delta_m,
                    )
                if self.skill == "GRASP" and not snapshot_has_physical_context(payload):
                    legacy_grasp_ids.append(selected)
        else:
            self.unwrapped.reset(env_ids=ids)
            if self.skill == "GRASP":
                legacy_grasp_ids.append(ids)
            else:
                self.gripper_action.prime_closed(ids)
        if legacy_grasp_ids:
            self._apply_profile_rest_heights(torch.cat(legacy_grasp_ids))
        self.steps[ids] = 0
        self.stable_count[ids] = 0
        self.prev_force[ids] = 0.0
        self.prev_unit[ids] = 0.0
        self.finished[ids] = False
        self.snapshot_recovery_active = True
        self.entrance_contact_recovered[ids] = False
        self.grasp_contact_seen[ids] = False
        self.entrance_height_recovery_required[ids] = False
        self.entrance_height_recovery_complete[ids] = False
        self.last_success[ids] = False
        self.last_failure[ids] = False
        self.last_timeout[ids] = False
        self.unwrapped.episode_length_buf[ids] = 0
        self.entry_red_z[ids] = self._measure()["red"][ids, 2]
        # A reset invalidates any pose estimate from the previous episode.
        self.visual_red = self.visual_blue = None

    def mark_continuous_handoff(self) -> None:
        """Reset policy-only entrance state for a no-reset skill switch."""
        self.snapshot_recovery_active = False
        self.prev_unit.zero_()
        if self.measured is not None:
            self.prev_force[:] = self.measured["force"]

    def _apply_profile_rest_heights(self, ids) -> None:
        """Place randomized-size GRASP objects on the table after any restore."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if not len(ids):
            return
        origins = self.scene.env_origins[ids]
        profiles = (
            (self.red, self.physical_batch.object_size_m[ids, 2]),
            (self.blue, self.physical_batch.support_size_m[ids, 2]),
        )
        for asset, height in profiles:
            position = to_torch(asset.data.root_pos_w)[ids].clone()
            position[:, 2] = origins[:, 2] + height / 2.0 + 0.0003
            orientation = to_torch(asset.data.root_quat_w)[ids].clone()
            asset.write_root_pose_to_sim_index(
                root_pose=torch.cat([position, orientation], dim=-1), env_ids=ids
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids
            )
        self.unwrapped.sim.forward()

    def set_visual_object_positions(self, red, blue) -> None:
        """Set (or clear) red/blue positions exposed to the policy."""
        if red is None or blue is None:
            if red is not None or blue is not None:
                raise ValueError("red and blue visual positions must be set together")
            self.visual_red = self.visual_blue = None
            return
        if self.measured is None:
            raise RuntimeError("set_visual_object_positions requires initialized measurements")
        red_t = torch.as_tensor(red, device=self.device, dtype=self.measured["red"].dtype)
        blue_t = torch.as_tensor(blue, device=self.device, dtype=self.measured["blue"].dtype)
        expected = (self.num_envs, 3)
        if red_t.shape != expected or blue_t.shape != expected:
            raise ValueError(f"visual positions must have shape {expected}")
        if not torch.isfinite(red_t).all() or not torch.isfinite(blue_t).all():
            raise ValueError("visual positions must be finite")
        self.visual_red = red_t.clone()
        self.visual_blue = blue_t.clone()

    def set_object_roles(self, object_asset_name: str, support_asset_name: str) -> None:
        """Switch policy roles without stepping or resetting simulator state.

        This is used by multi-relation tasks such as a three-object stack.  It
        changes only wrapper bookkeeping; every rigid body and robot state in
        Isaac remains untouched.
        """
        object_asset_name = str(object_asset_name)
        support_asset_name = str(support_asset_name)
        if object_asset_name == support_asset_name:
            raise ValueError("object and support assets must differ")
        try:
            object_asset = self.scene[object_asset_name]
            support_asset = self.scene[support_asset_name]
        except KeyError as exc:
            raise ValueError(f"unknown scene asset: {exc.args[0]!r}") from exc

        self.object_asset_name = object_asset_name
        self.support_asset_name = support_asset_name
        self.red = object_asset
        self.blue = support_asset
        self.visual_red = self.visual_blue = None
        self.measured = self._measure()
        self.entry_red_z[:] = self.measured["red"][:, 2]
        self.prev_force[:] = self.measured["force"]
        self.prev_unit.zero_()
        self.steps.zero_()
        self.stable_count.zero_()
        self.finished.zero_()
        self.last_success.zero_()
        self.last_failure.zero_()
        self.last_timeout.zero_()
        self.grasp_contact_seen.zero_()

    def update_lift_target_from_support(self) -> torch.Tensor:
        """Set per-environment clearance above the current support object.

        A support may already be part of a tower.  The manipulated object must
        reach the ALIGN entrance above that live support height before planar
        transport starts, otherwise it can strike and dislodge the tower.
        """
        if self.measured is None:
            raise RuntimeError("measurements must be initialized before LIFT")
        align_target = getattr(self, "align_target_height_m", self.align_height_target_m)
        required = (
            self.measured["blue"][:, 2]
            + align_target
            - self.entry_red_z
        )
        self.lift_target_height_m[:] = required.clamp_min(self.lift_height_m)
        return self.lift_target_height_m.clone()

    def _obs(self):
        m = self.measured
        force = m["force"]
        target = self.gripper_action.target_force_n.unsqueeze(-1).expand(-1, 2)
        force_rate = (force - self.prev_force) / max(float(self.unwrapped.step_dt), 1e-6)
        red_obs = m["red"] if self.visual_red is None else self.visual_red
        blue_obs = m["blue"] if self.visual_blue is None else self.visual_blue
        rel = red_obs - blue_obs
        gap = m["grip"].sum(dim=-1, keepdim=True)
        size = self.geometry_adapter.legacy_size_features()
        fields = [
            rel / 0.05,
            (m["ee"] - red_obs) / 0.05,
            m["vel"] / 0.2,
            m["angular"] / 2.0,
            m["red_quat"],
            m["blue_quat"],
            m["grip"] / 0.04,
            force / 10.0,
            m["q"] / 3.0,
            m["qd"] / 2.0,
            self.prev_unit,
            (self.stable_count / self.stable_steps).unsqueeze(-1),
            (self.steps / self.episode_steps).unsqueeze(-1),
            size,
            gap / 0.08,
            force.mean(dim=-1, keepdim=True) / 10.0,
            (force[:, :1] - force[:, 1:2]) / 10.0,
            force_rate / 20.0,
            self.gripper_action.target_force_n.unsqueeze(-1) / self.known_size.max_force_n,
            m["physical"].float().unsqueeze(-1),
        ]
        if self.include_physical_context:
            fields.append(self.physical_batch.action_context())
        obs = torch.cat(fields, dim=-1)
        if obs.shape != (self.num_envs, self.observation_dim) or not torch.isfinite(obs).all():
            raise RuntimeError("invalid known-size grasp observation")
        return TensorDict({"policy": obs.clamp(-10.0, 10.0)}, batch_size=[self.num_envs])

    def reset(self):
        self._restore(torch.arange(self.num_envs, device=self.device))
        self.measured = self._measure()
        self.entry_red_z[:] = self.measured["red"][:, 2]
        self.update_lift_target_from_support()
        self.prev_force[:] = self.measured["force"]
        return self._obs(), {}

    def get_observations(self):
        """Return the oracle observation rather than the base 94-D task state."""
        if self.measured is None:
            return self.reset()[0]
        return self._obs()

    def _reward(self, before, after, unit, requested_action, success, failure, timeout):
        force_error = (after["force"] - self.gripper_action.target_force_n.unsqueeze(-1)).abs().mean(-1)
        # The simulator's calibrated contact response is narrow (about
        # 4-8 N/finger for this object).  A broad exponential would make an
        # unreachable 9-25 N target almost as attractive as the reachable
        # prior, so use a sharper pressure-tracking signal for PPO credit.
        force_term = torch.exp(-force_error / 1.5)
        pressure_penalty = 0.05 * force_error
        balance_term = torch.exp(-(after["force"][:, 0] - after["force"][:, 1]).abs() / 2.0)
        physical_term = after["physical"].float()
        if self.skill == "LIFT":
            lift = (
                (after["red"][:, 2] - self.entry_red_z)
                / self.lift_target_height_m.clamp_min(1e-6)
            ).clamp(0, 1)
            progress = (
                8.0 * lift + physical_term + 0.8 * force_term
                + 0.1 * balance_term - pressure_penalty
            )
        elif self.skill == "TRANSPORT":
            xy_before = (before["red"][:, :2] - before["blue"][:, :2]).norm(dim=-1)
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            if self.transport_reward_variant == "absolute_v1":
                xy_progress = (1.0 - xy / 0.30).clamp(0, 1)
                transport_term = 5.0 * xy_progress
            else:
                # The original absolute-distance term gave almost the same
                # reward to a stationary policy and to a useful move.  Use a
                # bounded potential difference so PPO receives credit only
                # when the cube actually moves toward the target.  The small
                # goal potential keeps the final few centimetres learnable.
                xy_delta = (xy_before - xy).clamp(-0.05, 0.05)
                progress_term = 45.0 * xy_delta
                goal_term = (4.0 if self.transport_reward_variant == "progress_safety_v3" else 3.0) * torch.exp(-xy / 0.06)

                # The bounded reference controller moves the end effector in
                # the direction of the target cube.  This is a shaping signal,
                # not an action override: the policy still emits all five
                # values and physics determines the resulting cube motion.
                target_vec = after["blue"][:, :2] - after["ee"][:, :2]
                target_norm = target_vec.norm(dim=-1, keepdim=True).clamp_min(1e-5)
                command = unit[:, :2]
                command_norm = command.norm(dim=-1, keepdim=True)
                alignment = (command * target_vec / target_norm).sum(dim=-1)
                direction_term = 1.5 * alignment * command_norm.squeeze(-1)
                direction_term = torch.where(xy > self.transport_xy_m, direction_term,
                                             torch.zeros_like(direction_term))

                # Horizontal IK motion can make the held cube sag.  Penalize
                # only a meaningful drop from the settled entrance height;
                # small contact settling remains allowed.
                drop = (self.entry_red_z - after["red"][:, 2] - 0.005).clamp_min(0.0)
                if self.transport_reward_variant in (
                    "progress_height_hold_v4", "progress_settle_v5"
                ):
                    # Height hold must be bidirectional.  Penalizing only a
                    # downward drop lets an unconstrained policy drift upward
                    # until the fingers lose calibrated contact pressure.
                    height_error = (
                        after["red"][:, 2] - self.entry_red_z
                    ).abs()
                    height_term = -60.0 * (height_error - 0.004).clamp_min(0.0)
                else:
                    height_term = (-55.0 if self.transport_reward_variant == "progress_safety_v3" else -18.0) * drop
                if self.transport_reward_variant == "progress_safety_v3":
                    # Penalize unnecessary lateral command energy and reward
                    # closing when measured pressure is below the target. This
                    # reduces the large IK excursions seen on long diagonal
                    # transfers while preserving the learned five-value API.
                    action_cost = 0.18 * command_norm.squeeze(-1).square()
                    pressure_deficit = (
                        self.gripper_action.target_force_n - after["force"].mean(dim=-1)
                    ).clamp_min(0.0) / self.gripper_action.target_force_n.clamp_min(1e-5)
                    grip_hold = 0.6 * (-unit[:, 4]).clamp(0.0, 1.0) * pressure_deficit
                    pressure_bonus = 0.8 * self.gripper_action.pressure_tracking_ok(
                        after["force"]
                    ).float()
                    transport_term = (
                        progress_term + goal_term + direction_term + height_term
                        - action_cost + grip_hold + pressure_bonus
                    )
                else:
                    transport_term = progress_term + goal_term + direction_term + height_term
                if self.transport_reward_variant == "progress_settle_v5":
                    self.last_transport_reward_terms = transport_settle_reward_terms(
                        xy,
                        after["stability_speed"],
                        after["stability_angular_speed"],
                        unit,
                        cfg=self.transport_handoff_cfg,
                    )
                    settle = self.last_transport_reward_terms
                    transport_term = (
                        transport_term
                        + settle["settled_bonus"]
                        - settle["linear_speed_cost"]
                        - settle["angular_speed_cost"]
                        - settle["action_cost"]
                    )
            progress = (
                transport_term + physical_term + 0.8 * force_term
                + 0.1 * balance_term - pressure_penalty
            )
        elif self.skill == "ALIGN":
            xy_before = (before["red"][:, :2] - before["blue"][:, :2]).norm(dim=-1)
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            z_before = (before["red"][:, 2] - before["blue"][:, 2] - self.align_target_height_m).abs()
            z_error = (after["red"][:, 2] - after["blue"][:, 2] - self.align_target_height_m).abs()
            self.last_align_reward_terms = align_reward_terms(
                xy_before,
                xy,
                z_before,
                z_error,
                after["stability_speed"],
                after["stability_angular_speed"],
                object_upright_tilt_rad(before["red_quat"]),
                object_upright_tilt_rad(after["red_quat"]),
                unit,
                self.prev_unit,
                requested_action,
                after["left_height_error_m"],
                after["right_height_error_m"],
                self.physical_height_tolerance_m,
                cfg=self.align_reward_cfg,
            )
            terms = self.last_align_reward_terms
            progress = (
                terms["xy_progress"] + terms["xy_goal"] + terms["height_progress"]
                + physical_term + 0.8 * force_term + 0.1 * balance_term
                - pressure_penalty - terms["xy_margin_cost"] - terms["height_cost"]
                - terms["speed_cost"] - terms["action_cost"]
                - terms["saturation_cost"] - terms["grasp_margin_cost"]
                - terms["angular_speed_cost"] - terms["tilt_growth_cost"]
                - terms["action_delta_cost"] - terms["large_action_cost"]
            )
        elif self.skill == "DESCEND":
            xy_before = (before["red"][:, :2] - before["blue"][:, :2]).norm(dim=-1)
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            z_before = (before["red"][:, 2] - before["blue"][:, 2] - self.descend_target_height_m).abs()
            z_error = (after["red"][:, 2] - after["blue"][:, 2] - self.descend_target_height_m).abs()
            xy_progress = 35.0 * (xy_before - xy).clamp(-0.03, 0.03)
            height_progress = 30.0 * (z_before - z_error).clamp(-0.02, 0.02)
            goal_term = 5.0 * torch.exp(-xy / 0.012) * torch.exp(-z_error / 0.008)
            height_term = -40.0 * (z_error - self.descend_height_tolerance_m).clamp_min(0.0)
            speed_penalty = 0.5 * (
                after["stability_speed"] / self.descend_speed_mps
            ).clamp_min(0.0)
            progress = (
                xy_progress + height_progress + goal_term + height_term
                + physical_term + 0.8 * force_term + 0.1 * balance_term
                - pressure_penalty - speed_penalty
            )
        elif self.skill == "GRASP":
            before_error = grasp_contact_error(
                before["grasp_target"], before["left_tip"], before["right_tip"]
            )
            contact_error = grasp_contact_error(
                after["grasp_target"], after["left_tip"], after["right_tip"]
            )
            geometry_progress = 60.0 * (
                before_error.norm(dim=-1) - contact_error.norm(dim=-1)
            ).clamp(-0.02, 0.02)
            geometry_term = 3.0 * torch.exp(
                -contact_error[:, :2].norm(dim=-1) / 0.006
                -contact_error[:, 2].abs() / 0.004
            )
            arm_action_cost = 0.03 * unit[:, :4].square().sum(dim=-1)
            progress = (
                geometry_progress + geometry_term + 2.0 * physical_term
                + 0.8 * force_term + 0.1 * balance_term
                - pressure_penalty - arm_action_cost
            )
        else:
            progress = 2.0 * physical_term + 0.8 * force_term + 0.1 * balance_term - pressure_penalty
        terminal = 10.0 * success.float() - 10.0 * (failure | timeout).float()
        return progress - self.known_size.residual_action_penalty * unit[:, 4].square() + terminal

    def step(self, actions):
        if self.measured is None:
            self.reset()
        active = ~self.finished
        action = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if action.shape != (self.num_envs, 5):
            raise ValueError("known-size grasp policy action must be [N,5]")
        if self.arm_locked:
            action = action.clone()
            action[:, :4] = 0.0
        elif self.skill == "LIFT" and self.lift_vertical_only:
            # Initial LIFT curriculum: preserve the validated grasp geometry
            # and learn only vertical motion plus the pressure residual.
            action = action.clone()
            action[:, 0] = 0.0
            action[:, 1] = 0.0
            action[:, 3] = 0.0
        elif self.skill == "TRANSPORT" and self.transport_planar_only:
            action = action.clone()
            action[:, 2] = 0.0
            action[:, 3] = 0.0
        if self.skill == "ALIGN":
            relative = self.measured["red"] - self.measured["blue"]
            action = project_align_motion(
                action,
                relative,
                self.align_target_height_m,
                translation_limit_m=self.align_translation_limit_m,
                planar_height_margin_m=self.align_translation_limit_m,
            )
        elif self.skill == "DESCEND":
            relative = self.measured["red"] - self.measured["blue"]
            action = project_descend_motion(
                action,
                relative,
                self.descend_target_height_m,
                xy_tolerance_m=self.align_xy_m,
                inner_xy_m=self.align_inner_xy_m,
                translation_limit_m=self.descend_translation_limit_m,
            )
        if self.skill in ("GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND"):
            # All carrying skills own closing pressure only. Opening belongs
            # exclusively to RELEASE_STABILIZE; enforce the module boundary
            # before the residual is decoded.
            action = action.clone()
            action[:, 4].clamp_(max=-0.05)
        if self.skill == "GRASP":
            self.grasp_contact_seen = update_grasp_contact_seen(
                self.grasp_contact_seen,
                self.measured["force"],
                active,
                threshold_n=self.grasp_touch_force_threshold_n,
            )
        if self.skill == "GRASP" and self.grasp_profile.geometry == "box":
            yaw_error = parallel_jaw_yaw_error(
                self.measured["red_quat"],
                self.physical_batch.object_size_m,
                self.measured["left_tip"],
                self.measured["right_tip"],
            )
            contact_free = ~self.grasp_contact_seen
            action, _aligning = project_pregrasp_edge_alignment(
                action,
                yaw_error,
                contact_free & active,
                tolerance_rad=self.pregrasp_yaw_tolerance_rad,
            )
        translation_limit = (
            self.lift_translation_limit_m
            if self.skill == "LIFT"
            else self.transport_translation_limit_m
            if self.skill == "TRANSPORT"
            else self.align_translation_limit_m
            if self.skill == "ALIGN"
            else self.descend_translation_limit_m
            if self.skill == "DESCEND"
            else self.release_translation_limit_m
            if self.skill == "RELEASE_STABILIZE"
            else self.retreat_translation_limit_m
            if self.skill == "RETREAT"
            else 0.005
        )
        if self.prepress_steps:
            # Let contact pressure settle before any vertical acceleration.
            # This is a deterministic entrance conditioner, not a learned
            # replacement for the carrying skill policy.
            prepress = (self.steps < self.prepress_steps) & active
            if bool(prepress.any()):
                action = action.clone()
                action[prepress, :4] = 0.0
                action[prepress, 4] = self.prepress_grip
        if self.skill == "LIFT" and self.lift_arm_warmup_steps:
            warm = (self.steps < self.lift_arm_warmup_steps) & active
            if bool(warm.any()):
                action = action.clone()
                scale = entrance_action_scale(
                    self.steps,
                    warmup_steps=self.lift_arm_warmup_steps,
                    start_scale=self.lift_arm_warmup_scale,
                )
                action[warm, :4] *= scale[warm, None].to(action)
        if (
            self.snapshot_recovery_active
            and
            self.skill in ("TRANSPORT", "ALIGN", "DESCEND")
            and self.entrance_gravity_compensation_until_contact
        ):
            # A restored snapshot has poses and joint state but no contact
            # impulses.  When a stale saved servo target is rejected, wide or
            # asymmetric fingers may need more than the fixed prepress window
            # to close from their real joint positions.  Do not let the actor
            # move the wrist until the pressure loop has rebuilt a real grasp.
            pressure_ready = self.gripper_action.pressure_tracking_ok(
                self.measured["force"]
            )
            height_centered = entrance_fingertip_height_ready(
                self.measured["grasp_target"],
                self.measured["left_tip"],
                self.measured["right_tip"],
                tolerance_m=self.entrance_height_correction_tolerance_m,
            )
            self.entrance_height_recovery_required, self.entrance_height_recovery_complete = (
                update_entrance_height_recovery(
                    self.entrance_height_recovery_required,
                    self.entrance_height_recovery_complete,
                    self.measured["left_height_aligned"]
                    & self.measured["right_height_aligned"],
                    height_centered,
                    (self.steps > 0) & active,
                )
            )
            carry_contact_ready = entrance_grasp_recovery_ready(
                entrance_contact_ready(
                    self.measured["between_fingertips"],
                    self.measured["finger_a_contact"],
                    self.measured["finger_b_contact"],
                    pressure_ready,
                ),
                self.measured["left_height_aligned"],
                self.measured["right_height_aligned"],
                self.entrance_height_recovery_complete,
            )
            self.entrance_contact_recovered = update_entrance_contact_recovered(
                self.entrance_contact_recovered,
                carry_contact_ready & active,
            )
            contact_recovery = entrance_recovery_active(
                self.entrance_contact_recovered, active
            )
            if bool(contact_recovery.any()):
                action = action.clone()
                action[contact_recovery, :4] = 0.0
                height_recovery = (
                    contact_recovery
                    & self.entrance_height_recovery_required
                    & ~self.entrance_height_recovery_complete
                )
                height_step = entrance_fingertip_height_step(
                    self.measured["grasp_target"],
                    self.measured["left_tip"],
                    self.measured["right_tip"],
                    self.entrance_height_recovery_complete,
                    max_step_m=min(
                        self.entrance_height_correction_max_step_m,
                        translation_limit,
                    ),
                )
                action[height_recovery, 2] = (
                    height_step[height_recovery] / translation_limit
                )
                # A loaded closed grasp translates the payload with the wrist,
                # leaving the relative height error unchanged. Temporarily
                # release only misaligned entrances while gravity compensation
                # holds the payload, then close again at the corrected height.
                action[height_recovery, 4] = 1.0
        if (
            self.skill in ("GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND")
            and self.entrance_arm_warmup_steps
        ):
            # The restored post-transport pose may still contain contact
            # transients. Gradually hand arm control to the skill actor so its
            # first learned target cannot inject a full IK step into the hold.
            age = (self.steps - self.prepress_steps).clamp_min(0)
            warm = (
                (age < self.entrance_arm_warmup_steps)
                & self.entrance_contact_recovered
                & active
            )
            if bool(warm.any()):
                action = action.clone()
                scale = entrance_action_scale(
                    age,
                    warmup_steps=self.entrance_arm_warmup_steps,
                    start_scale=self.entrance_arm_warmup_scale,
                )
                action[warm, :4] *= scale[warm, None].to(action)
        if self.skill == "GRASP":
            # Freeze the wrist after the first measured fingertip touch while
            # the learned gripper command continues closing.  This prevents
            # subsequent arm targets from repeatedly striking and spinning
            # the object; physical success retains its separate 0.5 N gate.
            grasp_hold = active & self.grasp_contact_seen
            if bool(grasp_hold.any()):
                action = action.clone()
                action[grasp_hold, :4] = 0.0
        raw, unit = known_size_raw_action(action, translation_limit_m=translation_limit)
        before = self.measured
        if (
            self.skill in ("GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND")
            and self.carry_leveling_max_angle_rad > 0.0
        ):
            leveling = jaw_leveling_axis_angle(
                before["left_tip"],
                before["right_tip"],
                max_angle_rad=self.carry_leveling_max_angle_rad,
            )
            pressure_ready = self.gripper_action.pressure_tracking_ok(
                before["force"]
            )
            leveling_ready = (
                active
                & (self.steps >= self.prepress_steps)
                & before["physical"]
                & pressure_ready
            )
            if bool(leveling_ready.any()):
                age = (self.steps - self.prepress_steps).clamp_min(0)
                scale = entrance_action_scale(
                    age,
                    warmup_steps=self.entrance_arm_warmup_steps,
                    start_scale=self.entrance_arm_warmup_scale,
                ).to(raw)
                raw[leveling_ready, 3:5] = (
                    leveling[leveling_ready].to(raw)
                    * scale[leveling_ready, None]
                )
        contact_refresh = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if (
            self.snapshot_recovery_active
            and self.skill in ("TRANSPORT", "ALIGN", "DESCEND", "RELEASE_STABILIZE")
        ):
            contact_refresh = (
                self.steps < self.entrance_contact_refresh_steps
            ) & active
        self.gripper_action.set_position_hold_mask(contact_refresh)
        # PhysX scene-state snapshots do not contain contact impulses. During
        # downstream entrance conditioning, cancel only payload gravity while
        # the restored closed fingers rebuild real contacts. The wrench is
        # removed before the learned skill is allowed to move.
        composer = self.red.permanent_wrench_composer
        composer.reset()
        if (
            self.snapshot_recovery_active
            and
            self.skill in ("TRANSPORT", "ALIGN", "DESCEND", "RELEASE_STABILIZE")
            and self.entrance_gravity_compensation
        ):
            supported = (self.steps < self.prepress_steps) & active
            if self.skill in ("ALIGN", "DESCEND"):
                supported |= (
                    self.steps
                    < self.prepress_steps + self.entrance_arm_warmup_steps
                ) & active
            if self.entrance_gravity_compensation_until_contact:
                # Scene snapshots contain poses and servo targets, but not
                # contact impulses. Keep the payload from falling while the
                # restored fingers rebuild real geometry/pressure contact.
                pressure_ok_before = self.gripper_action.pressure_tracking_ok(
                    before["force"]
                )
                supported |= (
                    (~before["physical"] | ~pressure_ok_before)
                    & ~self.entrance_contact_recovered
                    & active
                )
            ids = supported.nonzero(as_tuple=False).flatten()
            if len(ids):
                damping_force = entrance_velocity_damping_force(
                    self.physical_batch.object_mass_kg,
                    before["vel"],
                    entrance_recovery_active(
                        self.entrance_contact_recovered, active
                    ),
                    step_dt_s=float(self.unwrapped.step_dt),
                    max_acceleration_mps2=(
                        ENTRANCE_RECOVERY_MAX_DAMPING_ACCELERATION_MPS2
                    ),
                )
                forces = damping_force[ids].unsqueeze(1)
                forces[:, 0, 2] += (
                    self.physical_batch.object_mass_kg[ids] * 9.81
                )
                composer.add_forces_and_torques(
                    forces,
                    torch.zeros_like(forces),
                    env_ids=ids,
                    is_global=True,
                )
        self.env.step(raw)
        if self.unwrapped.reset_buf.any():
            raise RuntimeError("base simulator reset during known-size grasp step")
        self.steps += active.long()
        after = self._measure()
        after["control_speed"] = (
            (after["red"] - before["red"]).norm(dim=-1)
            / max(float(self.unwrapped.step_dt), 1e-6)
        )
        after["stability_speed"] = stability_speed_from_source(
            after["speed"], after["control_speed"], self.stability_speed_source
        )
        after["control_angular_speed"] = quaternion_control_angular_speed(
            before["red_quat"],
            after["red_quat"],
            step_dt_s=max(float(self.unwrapped.step_dt), 1e-6),
        )
        after["instantaneous_angular_speed"] = after["angular"].norm(dim=-1)
        after["stability_angular_speed"] = stability_speed_from_source(
            after["instantaneous_angular_speed"],
            after["control_angular_speed"],
            self.stability_speed_source,
        )
        pressure_ok = self.gripper_action.pressure_tracking_ok(after["force"])
        if self.skill == "GRASP":
            ready = after["physical"] & pressure_ok
        elif self.skill == "LIFT":
            ready = (
                after["physical"]
                & pressure_ok
                & ((after["red"][:, 2] - self.entry_red_z) >= self.lift_target_height_m)
            )
        elif self.skill == "TRANSPORT":
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            ready = transport_handoff_ready(
                after["physical"],
                pressure_ok,
                xy,
                after["stability_speed"],
                after["stability_angular_speed"],
                cfg=self.transport_handoff_cfg,
            )
        elif self.skill == "ALIGN":
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            z_error = (
                after["red"][:, 2] - after["blue"][:, 2] - self.align_target_height_m
            ).abs()
            ready = (
                after["physical"] & pressure_ok & (xy <= self.align_xy_m)
                & (z_error <= self.align_height_tolerance_m)
                & (after["stability_speed"] <= self.align_speed_mps)
            )
        elif self.skill == "DESCEND":
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            z_error = (
                after["red"][:, 2] - after["blue"][:, 2] - self.descend_target_height_m
            ).abs()
            ready = (
                after["physical"] & pressure_ok & (xy <= self.align_xy_m)
                & (z_error <= self.descend_height_tolerance_m)
                & (after["stability_speed"] <= self.descend_speed_mps)
            )
        elif self.skill == "RELEASE_STABILIZE":
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            z_error = (
                after["red"][:, 2] - after["blue"][:, 2] - self.descend_target_height_m
            ).abs()
            ready = (
                after["open"] & (xy <= self.release_xy_m)
                & (z_error <= self.release_height_tolerance_m)
                & (after["stability_speed"] <= self.release_speed_mps)
            )
        else:
            xy = (after["red"][:, :2] - after["blue"][:, :2]).norm(dim=-1)
            z_error = (
                after["red"][:, 2] - after["blue"][:, 2] - self.descend_target_height_m
            ).abs()
            ee_red = after["ee"] - after["red"]
            ready = (
                after["open"] & (xy <= self.release_xy_m)
                & (z_error <= self.release_height_tolerance_m)
                & (ee_red.norm(dim=-1) >= self.retreat_distance_m)
                & (ee_red[:, 2] >= self.retreat_height_m)
                & (after["stability_speed"] <= self.retreat_speed_mps)
            )
        self.stable_count = torch.where(ready & active, self.stable_count + 1, torch.zeros_like(self.stable_count))
        success = (self.stable_count >= self.stable_steps) & active
        failure_guard = torch.maximum(
            torch.ones_like(self.steps),
            torch.full_like(self.steps, self.prepress_steps),
        )
        if self.skill in ("ALIGN", "DESCEND"):
            # Give the learned actor the same entrance transition budget as
            # the safety layer before treating a temporarily missing contact
            # impulse as an unrecoverable payload drop.
            failure_guard = torch.maximum(
                failure_guard,
                torch.full_like(
                    self.steps,
                    self.prepress_steps + self.entrance_arm_warmup_steps,
                ),
            )
        if self.skill in ("ALIGN", "DESCEND"):
            failure = carrying_grasp_lost(
                after["between_fingertips"],
                after["finger_a_contact"],
                after["finger_b_contact"],
            ) & (self.steps > failure_guard) & active
        elif self.skill in ("LIFT", "TRANSPORT"):
            failure = ((~after["physical"]) & (self.steps > failure_guard)) & active
        elif self.skill in ("RELEASE_STABILIZE", "RETREAT"):
            rel = after["red"] - after["blue"]
            failure = ((rel[:, 2] < 0.025) | (rel[:, :2].norm(dim=-1) > 0.15)) & active
        else:
            failure = torch.zeros_like(active)
        if (
            self.skill in ("TRANSPORT", "ALIGN", "DESCEND")
            and self.entrance_gravity_compensation_until_contact
        ):
            # Contact recovery is an explicit entrance phase.  Let it exhaust
            # the episode timeout instead of reporting a payload drop while
            # gravity compensation and the locked arm are still rebuilding
            # the grasp.
            height_centered_after = entrance_fingertip_height_ready(
                after["grasp_target"],
                after["left_tip"],
                after["right_tip"],
                tolerance_m=self.entrance_height_correction_tolerance_m,
            )
            self.entrance_height_recovery_required, self.entrance_height_recovery_complete = (
                update_entrance_height_recovery(
                    self.entrance_height_recovery_required,
                    self.entrance_height_recovery_complete,
                    after["left_height_aligned"] & after["right_height_aligned"],
                    height_centered_after,
                    active,
                )
            )
            carry_contact_ready = entrance_grasp_recovery_ready(
                entrance_contact_ready(
                    after["between_fingertips"],
                    after["finger_a_contact"],
                    after["finger_b_contact"],
                    pressure_ok,
                ),
                after["left_height_aligned"],
                after["right_height_aligned"],
                self.entrance_height_recovery_complete,
            )
            self.entrance_contact_recovered = update_entrance_contact_recovered(
                self.entrance_contact_recovered,
                carry_contact_ready & active,
            )
            entrance_recovery = entrance_recovery_active(
                self.entrance_contact_recovered, active
            )
            failure &= ~entrance_recovery
        timeout = (self.steps >= self.episode_steps) & ~success & ~failure & active
        done = success | failure | timeout
        self.last_success = success.clone()
        self.last_failure = failure.clone()
        self.last_timeout = timeout.clone()
        reward = self._reward(
            before, after, unit, action, success, failure, timeout
        )
        self.prev_unit[active] = unit[active]
        self.prev_force[active] = before["force"][active]
        self.measured = after
        self.finished |= done
        self.total_steps += self.num_envs
        ids = done.nonzero(as_tuple=False).flatten()
        self.completed += len(ids)
        self.successes += int(success.sum())
        self.failures += int(failure.sum())
        self.timeouts += int(timeout.sum())
        if self.auto_reset and len(ids):
            self._restore(ids)
            self.measured = self._measure()
        extras = {"log": {
            "known_size_grasp/reward": reward.mean(),
            "known_size_grasp/physical": after["physical"].float().mean(),
            "known_size_grasp/force_mean_n": after["force"].mean(),
            "known_size_grasp/target_force_n": self.gripper_action.target_force_n.mean(),
            "known_size_grasp/pressure_error_n": (
                after["force"] - self.gripper_action.target_force_n.unsqueeze(-1)
            ).abs().amax(dim=-1).mean(),
            "known_size_grasp/force_imbalance_n": (
                after["force"][:, 0] - after["force"][:, 1]
            ).abs().mean(),
            "known_size_grasp/pressure_ok": pressure_ok.float().mean(),
        }}
        if self.skill == "ALIGN":
            extras["log"].update({
                f"known_size_grasp/align_{name}": value.mean()
                for name, value in self.last_align_reward_terms.items()
            })
        if self.skill == "TRANSPORT" and self.last_transport_reward_terms:
            extras["log"].update({
                f"known_size_grasp/transport_{name}": value.mean()
                for name, value in self.last_transport_reward_terms.items()
            })
        return self._obs(), reward, done, extras

    def statistics(self):
        return {
            "skill": self.skill,
            "known_size": asdict(self.known_size),
            "arm_locked": self.arm_locked,
            "lift_vertical_only": self.lift_vertical_only,
            "lift_translation_limit_m": self.lift_translation_limit_m,
            "lift_prepress_steps": self.lift_prepress_steps,
            "lift_prepress_grip": self.lift_prepress_grip,
            "entrance_prepress_steps": self.entrance_prepress_steps,
            "entrance_prepress_grip": self.entrance_prepress_grip,
            "entrance_gravity_compensation": self.entrance_gravity_compensation,
            "entrance_gravity_compensation_until_contact": (
                self.entrance_gravity_compensation_until_contact
            ),
            "entrance_contact_refresh_steps": self.entrance_contact_refresh_steps,
            "entrance_servo_target_blend": self.entrance_servo_target_blend,
            "entrance_servo_target_max_delta_m": (
                self.entrance_servo_target_max_delta_m
            ),
            "lift_arm_warmup_steps": self.lift_arm_warmup_steps,
            "lift_arm_warmup_scale": self.lift_arm_warmup_scale,
            "carry_leveling_max_angle_rad": self.carry_leveling_max_angle_rad,
            "pregrasp_yaw_tolerance_rad": self.pregrasp_yaw_tolerance_rad,
            "grasp_touch_force_threshold_n": self.grasp_touch_force_threshold_n,
            "transport_planar_only": self.transport_planar_only,
            "transport_reward_variant": self.transport_reward_variant,
            "transport_translation_limit_m": self.transport_translation_limit_m,
            "transport_xy_m": self.transport_xy_m,
            "transport_speed_mps": self.transport_speed_mps,
            "transport_angular_speed_radps": self.transport_angular_speed_radps,
            "transport_handoff": self.transport_handoff_cfg.as_dict(),
            "align_xy_m": self.align_xy_m,
            "align_inner_xy_m": self.align_inner_xy_m,
            "align_reward": asdict(self.align_reward_cfg),
            "align_height_target_m": self.align_height_target_m,
            "align_height_tolerance_m": self.align_height_tolerance_m,
            "align_speed_mps": self.align_speed_mps,
            "align_angular_speed_radps": self.align_angular_speed_radps,
            "align_upright_tolerance_rad": self.align_upright_tolerance_rad,
            "align_translation_limit_m": self.align_translation_limit_m,
            "descend_height_target_m": self.descend_height_target_m,
            "descend_target_mode": self.descend_target_mode,
            "descend_target_range_m": [
                float(self.descend_target_height_m.min()),
                float(self.descend_target_height_m.max()),
            ],
            "descend_height_tolerance_m": self.descend_height_tolerance_m,
            "descend_speed_mps": self.descend_speed_mps,
            "descend_translation_limit_m": self.descend_translation_limit_m,
            "observation_dim": self.observation_dim,
            "physical_context_version": (
                self.physical_batch.context_version if self.include_physical_context else None
            ),
            "geometry_bundle": self.physical_batch.geometry_bundle,
            "stability_speed_source": self.stability_speed_source,
            "action_dim": 5,
            "completed_episodes": self.completed,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
        }
