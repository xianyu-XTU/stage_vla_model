"""Continuous, size-conditioned gripper action for the v5 oracle phase.

The public action remains one scalar for the gripper.  Positive values open
the fingers; negative values close them, with the magnitude selecting a
normalized pressure target.  The action term itself owns only the low-level
position target.  It never emits raw torques.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from stage_vla.rl.known_size_grasp import (
    KnownSizeGraspConfig,
)
from stage_vla.rl.object_physics import GeometryAdapter, LoadAdapter, PhysicalObjectBatch
from .state_readers import net_force_per_env, to_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class KnownSizeGraspActionCfg(ActionTermCfg):
    """Configuration for :class:`KnownSizeGraspAction`."""

    class_type: type["KnownSizeGraspAction"] | str = (
        "{DIR}.known_size_grasp_action:KnownSizeGraspAction"
    )
    joint_names: list[str] = MISSING
    width_m: float = MISSING
    depth_m: float = MISSING
    height_m: float = MISSING
    mass_kg: float = MISSING
    grasp_width_ratio: float = 1.0
    jaw_clearance_m: float = 0.002
    max_compression_m: float = 0.004
    friction_coefficient: float = 0.4
    safety_factor: float = 2.0
    min_force_n: float = 5.0
    max_force_n: float = 40.0
    pressure_tolerance_n: float = 1.5
    force_balance_tolerance_n: float = 1.5
    lift_acceleration_mps2: float = 0.5
    joint_min_m: float = 0.0
    joint_max_m: float = 0.04
    open_position_m: float = 0.04
    left_sensor_name: str = "left_finger_contact"
    right_sensor_name: str = "right_finger_contact"
    gain_m_per_n: float = 0.0002
    max_step_m: float = 0.001
    residual_force_range_n: float = 4.0
    residual_deadband: float = 0.0


class KnownSizeGraspAction(ActionTerm):
    """Apply bounded joint-position feedback from known size and contact force."""

    cfg: KnownSizeGraspActionCfg

    def __init__(self, cfg: KnownSizeGraspActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        if len(self._joint_ids) != 2:
            raise ValueError("known-size grasp action requires exactly two gripper joints")
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._target_force_n = torch.zeros(self.num_envs, device=self.device)
        self._last_measured_force_n = torch.zeros(self.num_envs, 2, device=self.device)
        self._position_hold_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._cfg = KnownSizeGraspConfig(
            width_m=cfg.width_m,
            depth_m=cfg.depth_m,
            height_m=cfg.height_m,
            mass_kg=cfg.mass_kg,
            grasp_width_m=cfg.width_m * cfg.grasp_width_ratio,
            jaw_clearance_m=cfg.jaw_clearance_m,
            max_compression_m=cfg.max_compression_m,
            friction_coefficient=cfg.friction_coefficient,
            safety_factor=cfg.safety_factor,
            min_force_n=cfg.min_force_n,
            max_force_n=cfg.max_force_n,
            pressure_tolerance_n=cfg.pressure_tolerance_n,
            force_balance_tolerance_n=cfg.force_balance_tolerance_n,
            residual_force_range_n=cfg.residual_force_range_n,
            lift_acceleration_mps2=cfg.lift_acceleration_mps2,
            joint_min_m=cfg.joint_min_m,
            joint_max_m=cfg.joint_max_m,
        )
        self._cfg.validate()
        self._grasp_width_ratio = float(cfg.grasp_width_ratio)
        if not 0 < cfg.open_position_m <= cfg.joint_max_m + 1e-9:
            raise ValueError("open_position_m must lie in (0, joint_max_m]")
        if cfg.gain_m_per_n <= 0 or cfg.max_step_m <= 0 or cfg.residual_force_range_n < 0:
            raise ValueError("feedback gains and residual force range must be valid")
        self._left_sensor = env.scene[cfg.left_sensor_name]
        self._right_sensor = env.scene[cfg.right_sensor_name]
        self._open_position_m = float(cfg.open_position_m)
        self.set_physical_batch(
            PhysicalObjectBatch.from_scalar_config(self._cfg, self.num_envs)
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def target_force_n(self) -> torch.Tensor:
        return self._target_force_n.clone()

    @property
    def measured_force_n(self) -> torch.Tensor:
        return self._last_measured_force_n.clone()

    @property
    def oracle_config(self) -> KnownSizeGraspConfig:
        return self._cfg

    @property
    def physical_batch(self) -> PhysicalObjectBatch:
        return self._physical_batch

    @property
    def geometry_adapter(self) -> GeometryAdapter:
        return self._geometry_adapter

    @property
    def load_adapter(self) -> LoadAdapter:
        return self._load_adapter

    def set_physical_batch(self, physical: PhysicalObjectBatch) -> None:
        """Install the exact per-environment profile used by the simulator."""
        if not isinstance(physical, PhysicalObjectBatch):
            raise TypeError("physical must be a PhysicalObjectBatch")
        if physical.num_envs != self.num_envs:
            raise ValueError("physical batch size must equal num_envs")
        self._physical_batch = physical.to(self.device)
        self._geometry_adapter = GeometryAdapter(
            self._physical_batch,
            grasp_width_ratio=self._grasp_width_ratio,
            jaw_clearance_m=self._cfg.jaw_clearance_m,
            max_compression_m=self._cfg.max_compression_m,
            joint_min_m=self._cfg.joint_min_m,
            joint_max_m=self._cfg.joint_max_m,
        )
        self._load_adapter = LoadAdapter(
            self._physical_batch,
            friction_coefficient=self._cfg.friction_coefficient,
            safety_factor=self._cfg.safety_factor,
            lift_acceleration_mps2=self._cfg.lift_acceleration_mps2,
            min_force_n=self._cfg.min_force_n,
            max_force_n=self._cfg.max_force_n,
            residual_force_range_n=self._cfg.residual_force_range_n,
            residual_deadband=float(self.cfg.residual_deadband),
            pressure_tolerance_n=self._cfg.pressure_tolerance_n,
            force_balance_tolerance_n=self._cfg.force_balance_tolerance_n,
        )
        self._processed_actions[:] = self._open_position_m
        self._target_force_n[:] = self._load_adapter.initial_force_target_n

    def pressure_tracking_ok(self, measured_force_n: torch.Tensor) -> torch.Tensor:
        return self._load_adapter.tracking_ok(measured_force_n, self._target_force_n)

    def set_position_hold_mask(self, mask: torch.Tensor) -> None:
        """Hold the current servo target while restored contacts refresh."""
        value = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if value.shape != (self.num_envs,):
            raise ValueError("position hold mask must have shape [num_envs]")
        self._position_hold_mask[:] = value

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 1) or not torch.isfinite(actions).all():
            raise ValueError("known-size gripper action must be finite with shape [N,1]")
        self._raw_actions[:] = actions
        grip = actions[:, 0].clamp(-1.0, 1.0)
        measured = torch.stack(
            [net_force_per_env(self._left_sensor), net_force_per_env(self._right_sensor)], dim=-1
        ).to(device=self.device, dtype=self._processed_actions.dtype)
        # Keep a stateful position target.  Re-reading the physical joint
        # position every frame would reset the pressure integrator whenever
        # the servo lags behind its target, preventing cumulative squeeze.
        current = self._processed_actions.clone()
        self._last_measured_force_n[:] = measured
        target_force = self._load_adapter.target_from_grip(grip)
        self._target_force_n[:] = target_force
        open_mask = grip > 0
        close_target = self._geometry_adapter.pressure_feedback_step(
            current,
            measured,
            target_force,
            gain_m_per_n=float(self.cfg.gain_m_per_n),
            max_step_m=float(self.cfg.max_step_m),
        )
        open_target = torch.full_like(close_target, self._open_position_m)
        next_target = torch.where(open_mask[:, None], open_target, close_target)
        self._processed_actions[:] = torch.where(
            self._position_hold_mask[:, None], current, next_target
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._open_position_m
        self._target_force_n[env_ids] = self._load_adapter.initial_force_target_n[env_ids]
        self._last_measured_force_n[env_ids] = 0.0
        self._position_hold_mask[env_ids] = False

    def prime_closed(
        self,
        env_ids=None,
        *,
        joint_position=None,
        servo_target=None,
        target_blend: float = 1.0,
        max_target_delta_m: float | None = None,
    ) -> None:
        """Initialize the stateful servo from a restored closed joint pose.

        Restored post-grasp snapshots already contain a closed physical pose,
        but Isaac's action term does not serialize its internal target.  This
        hook prevents the first post-restore frame from reopening the fingers
        while a downstream skill is re-establishing contact.  Reusing the
        measured joint pose is essential for randomized widths: forcing every
        object to one size-derived compression limit can create an immediate
        squeeze before force feedback gets a chance to react.
        """
        if env_ids is None:
            env_ids = slice(None)
        if joint_position is None:
            joint_pos = to_torch(self._asset.data.joint_pos)[env_ids]
        else:
            joint_pos = torch.as_tensor(
                joint_position,
                device=self.device,
                dtype=self._processed_actions.dtype,
            )
            if joint_pos.ndim != 2:
                raise ValueError("restored joint_position must be a rank-2 tensor")
        restored = (
            joint_pos.clone()
            if joint_pos.shape[1] == len(self._joint_ids)
            else joint_pos[:, self._joint_ids].clone()
        )
        expected_rows = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        if restored.shape != (expected_rows, len(self._joint_ids)):
            raise ValueError("restored joint_position does not match selected environments")
        if not 0.0 <= float(target_blend) <= 1.0:
            raise ValueError("target_blend must lie in [0,1]")
        if max_target_delta_m is not None and (
            not torch.isfinite(torch.tensor(float(max_target_delta_m)))
            or float(max_target_delta_m) <= 0.0
        ):
            raise ValueError("max_target_delta_m must be positive and finite")
        if servo_target is not None:
            servo = torch.as_tensor(
                servo_target,
                device=self.device,
                dtype=self._processed_actions.dtype,
            )
            if servo.shape != restored.shape:
                raise ValueError("servo_target must match restored gripper joint positions")
            blended = torch.lerp(restored, servo, float(target_blend))
            if max_target_delta_m is not None:
                valid = (servo - restored).abs().amax(dim=-1) <= float(
                    max_target_delta_m
                )
                lower = self._geometry_adapter.compression_joint_min_m.to(
                    restored
                )
                if not isinstance(env_ids, slice):
                    lower = lower[env_ids]
                lower = lower.unsqueeze(-1)
                # A stale integral target must not be reused, but starting at
                # the wide measured pose can leave a restored payload without
                # contact for dozens of frames.  Rebuild from simulator truth
                # with one bounded closing preload, capped by the per-object
                # geometry limit.  Subsequent frames remain force-controlled.
                recovery = torch.maximum(
                    restored - float(max_target_delta_m), lower
                )
                blended = torch.where(valid[:, None], blended, recovery)
            restored = blended
        restored.clamp_(float(self._cfg.joint_min_m), float(self._cfg.joint_max_m))
        self._raw_actions[env_ids] = -float(self.cfg.residual_deadband)
        self._processed_actions[env_ids] = restored
        self._target_force_n[env_ids] = self._load_adapter.initial_force_target_n[env_ids]
        self._last_measured_force_n[env_ids] = 0.0
        self._position_hold_mask[env_ids] = False
