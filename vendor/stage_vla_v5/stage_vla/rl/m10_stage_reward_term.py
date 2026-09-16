"""Stateful M10 stage-aware RewardManager term.

M8 validated the stage separator and potential reward as a read-only observer on
an M7 scripted truth trajectory.  M10 turns that exact state machine into one
stateful Isaac Lab RewardManager term so the previous-potential update happens
exactly once per environment step.

The term intentionally owns all temporal state required by stage-aware reward:
M5 stable-grasp streak, M6 lift reference, task history, M7 success streak,
stage progression, and previous potential.  It must therefore never be split
into several reward terms with duplicated side effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import ManagerTermBase

from stage_vla.rl.action_dsl import M9B_CLOSE_TOKEN, M9B_KEEP_TOKEN, M9B_OPEN_TOKEN
from stage_vla.rl.baseline_reward import gripper_closedness
from stage_vla.rl.m10_release_credit import M10ReleaseCreditConfig, M10ReleaseCreditTracker
from stage_vla.rl.m10_settle_shaping import (
    M10_H2S2_OPEN_GATE_SEMANTICS,
    M10_H2S2_SETTLE_SHAPING_SEMANTICS,
    M10SettleShapingConfig,
    M10SettleShapingTracker,
)

from stage_vla.envs.state_readers import frame_positions_w, read_grasp_state, read_placement_state, to_torch
from stage_vla.stages import (
    LiftConfig,
    ManipulationStage,
    PhysicalGraspConfig,
    RedOnBlueConfig,
    RedOnBlueSuccessConfig,
    RedOnBlueSuccessTracker,
    StableGraspConfig,
    StableGraspTracker,
    StageAwareRewardConfig,
    StageAwareRewardTracker,
    StagePotentialConfig,
    StagePotentialInputs,
    StageProgressConfig,
    StageProgressTracker,
    TaskHistoryTracker,
    lift_diagnostics,
    physical_grasp_diagnostics,
    red_on_blue_diagnostics,
    stage_potential,
)


class M10StageAwareRewardTerm(ManagerTermBase):
    """Single-call-per-step M8 reward implementation for PPO training."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        # Isaac Lab 3.0 validates callable-class RewardTermCfg.params against
        # this class's __call__ signature before instantiation.  Keep one
        # manager-visible parameter (m10_params) and unpack it only here.
        p = cfg.params.get("m10_params")
        if not isinstance(p, dict):
            raise TypeError("M10StageAwareRewardTerm expects cfg.params['m10_params'] to be a dict")
        for name, value in p.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"M10 stage parameter {name!r} must be primitive int/float, "
                    f"got {type(value).__name__}"
                )

        self.physical_cfg = PhysicalGraspConfig(
            radial_tolerance_m=float(p["radial_tolerance_m"]),
            height_tolerance_m=float(p["grasp_height_tolerance_m"]),
            contact_force_threshold_n=float(p["contact_force_threshold_n"]),
            endpoint_margin=float(p["endpoint_margin"]),
        )
        self.stable_cfg = StableGraspConfig(int(p["required_stable_grasp_steps"]))
        self.lift_cfg = LiftConfig(float(p["minimum_object_lift_delta_m"]))
        self.placement_cfg = RedOnBlueConfig(
            xy_tolerance_m=float(p["xy_tolerance_m"]),
            target_height_diff_m=float(p["target_height_diff_m"]),
            height_tolerance_m=float(p["placement_height_tolerance_m"]),
            max_red_linear_speed_mps=float(p["max_red_linear_speed_mps"]),
            max_red_angular_speed_radps=float(p["max_red_angular_speed_radps"]),
            max_blue_linear_speed_mps=float(p["max_blue_linear_speed_mps"]),
            max_blue_angular_speed_radps=float(p["max_blue_angular_speed_radps"]),
        )
        self.success_cfg = RedOnBlueSuccessConfig(int(p["required_settle_steps"]))
        self.gripper_open_tolerance_m = float(p["gripper_open_tolerance_m"])
        self.cube_size_m = float(p["cube_size_m"])
        # 1 = DSL policy GRIP token supplied by the action wrapper (default).
        # 0 = continuous PPO: derive the release command from the actual gripper
        #     state instead, so the stage-aware reward works with continuous 4-D
        #     actions (M9A wrapper) without a discrete grip token.
        self.use_policy_gripper_token = bool(int(p.get("use_policy_gripper_token", 1)))
        if self.gripper_open_tolerance_m < 0:
            raise ValueError("gripper_open_tolerance_m must be >= 0")
        if self.cube_size_m <= 0:
            raise ValueError("cube_size_m must be > 0")

        self.reward_cfg = StageAwareRewardConfig(
            gamma=float(p["stage_gamma"]),
            stage_transition_bonus=float(p["stage_transition_bonus"]),
            success_bonus=float(p["success_bonus"]),
        )
        self.reward_cfg.validate()

        self.release_closed_joint_pos_m = float(p["release_closed_joint_pos_m"])
        self.release_credit_cfg = M10ReleaseCreditConfig(
            openness_progress_bonus=float(p["release_openness_progress_bonus"]),
            release_event_bonus=float(p["release_event_bonus"]),
            open_token_bonus=float(p["release_open_token_bonus"]),
        )
        self.release_credit_cfg.validate()
        self.settle_shaping_cfg = M10SettleShapingConfig(
            weight=float(p["settle_shaping_weight"]),
            gamma=float(p["stage_gamma"]),
            speed_ratio_cap=float(p["settle_speed_ratio_cap"]),
            max_red_linear_speed_mps=self.placement_cfg.max_red_linear_speed_mps,
            max_red_angular_speed_radps=self.placement_cfg.max_red_angular_speed_radps,
            max_blue_linear_speed_mps=self.placement_cfg.max_blue_linear_speed_mps,
            max_blue_angular_speed_radps=self.placement_cfg.max_blue_angular_speed_radps,
        )
        self.settle_shaping_cfg.validate()

        # ManagerTermBase already exposes ``num_envs`` and ``device`` as
        # read-only properties backed by ``self._env``.  Do not shadow or
        # assign them: Isaac Lab 3.0 raises AttributeError because those
        # properties intentionally have no setters.
        self.stable_tracker = StableGraspTracker(
            num_envs=self.num_envs, device=self.device, cfg=self.stable_cfg
        )
        self.history = TaskHistoryTracker(num_envs=self.num_envs, device=self.device)
        self.success_tracker = RedOnBlueSuccessTracker(
            num_envs=self.num_envs, device=self.device, cfg=self.success_cfg
        )
        self.stage_tracker = StageProgressTracker(
            num_envs=self.num_envs,
            device=self.device,
            cfg=StageProgressConfig(
                reach_transition_distance_m=self.cube_size_m,
                place_transition_distance_m=self.cube_size_m,
            ),
        )
        self.potential_cfg = StagePotentialConfig(
            object_size_m=self.cube_size_m,
            lift_scale_m=self.lift_cfg.minimum_object_lift_delta_m,
        )
        self.reward_tracker = StageAwareRewardTracker(
            num_envs=self.num_envs, device=self.device, cfg=self.reward_cfg
        )
        self.release_credit_tracker = M10ReleaseCreditTracker(
            num_envs=self.num_envs, device=self.device, cfg=self.release_credit_cfg
        )
        self.settle_shaping_tracker = M10SettleShapingTracker(
            num_envs=self.num_envs, device=self.device, cfg=self.settle_shaping_cfg
        )

        self._reference_red_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._has_reference_red_z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._transport_scale_m = torch.full(
            (self.num_envs,), self.cube_size_m, dtype=torch.float32, device=self.device
        )

        self.call_count = 0
        self.last_physical_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_stable_grasp = torch.zeros_like(self.last_physical_grasp)
        self.last_lifted = torch.zeros_like(self.last_physical_grasp)
        self.last_geometry_ok = torch.zeros_like(self.last_physical_grasp)
        self.last_settled = torch.zeros_like(self.last_physical_grasp)
        self.last_success_candidate = torch.zeros_like(self.last_physical_grasp)
        self.last_total_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_shaping_reward = torch.zeros_like(self.last_total_reward)
        self.last_transition_bonus = torch.zeros_like(self.last_total_reward)
        self.last_success_bonus = torch.zeros_like(self.last_total_reward)
        self.last_release_credit = torch.zeros_like(self.last_total_reward)
        self.last_open_token_credit = torch.zeros_like(self.last_total_reward)
        self.last_release_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_open_command_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_open_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_open_command_credited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_just_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_settle_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_settle_shaping_reward = torch.zeros_like(self.last_settle_potential)
        self.last_settle_speed_ratio = torch.zeros_like(self.last_settle_potential)
        # The M10 wrapper writes the current policy gripper token before env.step.
        # -1 means no token has been supplied/consumed yet.
        self._pending_policy_gripper_token = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._pending_open_command_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_policy_gripper_token = torch.full_like(self._pending_policy_gripper_token, -1)
        # Public diagnostic marker used by ENV/D2 guards.
        self.open_command_gate_semantics = M10_H2S2_OPEN_GATE_SEMANTICS
        self.settle_shaping_semantics = M10_H2S2_SETTLE_SHAPING_SEMANTICS

    @property
    def stage(self) -> torch.Tensor:
        return self.stage_tracker.stage

    @property
    def transport_scale_m(self) -> torch.Tensor:
        return self._transport_scale_m.clone()

    def set_policy_gripper_token(
        self,
        token: torch.Tensor,
        *,
        release_ready_before_action: torch.Tensor | None = None,
    ) -> None:
        """Supply the categorical GRIP token and its pre-action readiness gate.

        M10-8-T1 fixed the direct action-credit timing discovered by D2.
        M10-11-H4X preserves that successful T1 gate exactly: the wrapper
        supplies base ``release_ready(s_t)``. The historical H2-S2 tracker is
        retained only for diagnostics with training weight zero.

        ``release_ready_before_action=None`` falls back to the term's current
        ``last_release_ready`` to keep this helper safe for direct callers.
        """
        value = torch.as_tensor(token, device=self.device)
        if value.shape != (self.num_envs,):
            raise ValueError(
                f"gripper token must have shape ({self.num_envs},), got {tuple(value.shape)}"
            )
        if value.dtype.is_floating_point:
            rounded = torch.round(value)
            if not torch.allclose(value, rounded, atol=1.0e-6, rtol=0.0):
                raise ValueError("gripper token must be integer-valued")
            value = rounded.to(torch.long)
        else:
            value = value.to(torch.long)
        valid = (value == M9B_OPEN_TOKEN) | (value == M9B_KEEP_TOKEN) | (value == M9B_CLOSE_TOKEN)
        if not bool(valid.all().item()):
            raise ValueError("gripper token must be OPEN(0), KEEP(1), or CLOSE(2)")

        if release_ready_before_action is None:
            ready = self.last_release_ready
        else:
            ready = torch.as_tensor(release_ready_before_action, device=self.device)
        if ready.shape != (self.num_envs,):
            raise ValueError(
                "release_ready_before_action must have shape "
                f"({self.num_envs},), got {tuple(ready.shape)}"
            )
        if ready.dtype is not torch.bool:
            raise TypeError("release_ready_before_action must be torch.bool")

        self._pending_policy_gripper_token.copy_(value)
        self._pending_open_command_ready.copy_(ready)

    def _consume_policy_gripper_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        token = self._pending_policy_gripper_token.clone()
        if bool((token < 0).any().item()):
            raise RuntimeError(
                "M10 reward term was called without a policy GRIP token for one or more envs"
            )
        command_ready = self._pending_open_command_ready.clone()
        self._pending_policy_gripper_token.fill_(-1)
        self._pending_open_command_ready.fill_(False)
        self.last_policy_gripper_token = token.clone()
        self.last_open_command_ready = command_ready.clone()
        return token, command_ready

    def _env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be 1-D")
        if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= self.num_envs)):
            raise IndexError("env_ids out of range")
        return ids

    def _goal_from_positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        red = to_torch(self._env.scene["cube_2"].data.root_pos_w)[..., :3]
        blue = to_torch(self._env.scene["cube_1"].data.root_pos_w)[..., :3]
        goal = blue.clone()
        goal[:, 2] = blue[:, 2] + float(self.placement_cfg.target_height_diff_m)
        return red, blue, goal

    def _reach_distance(self, red_pos_w: torch.Tensor) -> torch.Tensor:
        indices, positions = frame_positions_w(self._env.scene["ee_frame"])
        ee = positions[:, indices.end_effector, :3]
        return torch.linalg.vector_norm(ee - red_pos_w, dim=-1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Reset temporal state and seed Phi(s_0) for selected environments."""
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return

        self.stable_tracker.reset(ids)
        self.history.reset(ids)
        self.success_tracker.reset(ids)
        self.stage_tracker.reset(ids)
        self._reference_red_z[ids] = 0.0
        self._has_reference_red_z[ids] = False

        red, _blue, goal = self._goal_from_positions()
        goal_distance = torch.linalg.vector_norm(red - goal, dim=-1).to(torch.float32)
        transport_scale = torch.maximum(
            goal_distance,
            torch.full_like(goal_distance, self.cube_size_m),
        )
        self._transport_scale_m[ids] = transport_scale[ids]

        reach_distance = self._reach_distance(red).to(torch.float32)
        # Every reset starts in REACH.  We can seed its potential without
        # touching ContactSensor data, which may not have a refreshed first
        # post-reset contact frame yet.
        stage = self.stage_tracker.stage
        zeros = torch.zeros_like(reach_distance)
        lift_residual = torch.full_like(
            reach_distance, float(self.lift_cfg.minimum_object_lift_delta_m)
        )
        phi = stage_potential(
            stage,
            StagePotentialInputs(
                reach_distance_m=reach_distance,
                grasp_error_m=zeros,
                lift_residual_m=lift_residual,
                transport_distance_m=goal_distance,
                place_distance_m=goal_distance,
                transport_scale_m=self._transport_scale_m,
            ),
            cfg=self.potential_cfg,
        )
        self.reward_tracker.reset(phi[ids], env_ids=ids)
        self.release_credit_tracker.reset(ids)
        self.settle_shaping_tracker.reset(ids)

        self.last_physical_grasp[ids] = False
        self.last_stable_grasp[ids] = False
        self.last_lifted[ids] = False
        self.last_geometry_ok[ids] = False
        self.last_settled[ids] = False
        self.last_success_candidate[ids] = False
        self.last_total_reward[ids] = 0.0
        self.last_shaping_reward[ids] = 0.0
        self.last_transition_bonus[ids] = 0.0
        self.last_success_bonus[ids] = 0.0
        self.last_release_credit[ids] = 0.0
        self.last_open_token_credit[ids] = 0.0
        self.last_release_ready[ids] = False
        self.last_open_command_ready[ids] = False
        self.last_open_command[ids] = False
        self.last_open_command_credited[ids] = False
        self.last_just_released[ids] = False
        self.last_settle_potential[ids] = 0.0
        self.last_settle_shaping_reward[ids] = 0.0
        self.last_settle_speed_ratio[ids] = 0.0
        self._pending_policy_gripper_token[ids] = -1
        self._pending_open_command_ready[ids] = False
        self.last_policy_gripper_token[ids] = -1

    @staticmethod
    def _grasp_error(m4) -> torch.Tensor:
        mean_height_error = 0.5 * (m4.left_height_error_m + m4.right_height_error_m)
        return torch.sqrt(m4.radial_error_m.square() + mean_height_error.square())

    def _write_step_logs(
        self, env, reward, release_credit, settle_shaping, stage, m4, stable, lifted, placement, success
    ) -> None:
        # RSL-RL consumes extras["log"] when present.  These are diagnostics,
        # not additional rewards.  Reset steps may overwrite this dictionary in
        # ManagerBasedRLEnv._reset_idx; losing one logging sample per episode is
        # acceptable and does not affect training state.
        if not isinstance(getattr(env, "extras", None), dict):
            return
        log = env.extras.setdefault("log", {})
        log["/M10/shaping_reward"] = reward.shaping_reward.mean()
        log["/M10/transition_bonus"] = reward.transition_bonus.mean()
        log["/M10/success_bonus"] = reward.sparse_success_reward.mean()
        log["/M10/release_credit"] = release_credit.total_reward.mean()
        log["/M10/settle_shaping_reward"] = settle_shaping.shaping_reward.mean()
        log["/M10/settle_potential"] = settle_shaping.potential.mean()
        log["/M10/settle_score_at_release_ready"] = (
            settle_shaping.settle_score * settle_shaping.release_ready.float()
        ).mean()
        log["/M10/open_token_credit"] = release_credit.command_reward.mean()
        log["/M10/open_token_at_release_ready_fraction"] = (
            release_credit.open_command & release_credit.release_ready
        ).float().mean()
        log["/M10/pre_action_release_ready_fraction"] = (
            release_credit.open_command_ready.float().mean()
        )
        log["/M10/open_token_at_pre_action_ready_fraction"] = (
            release_credit.open_command & release_credit.open_command_ready
        ).float().mean()
        log["/M10/just_open_token_credited_fraction"] = (
            release_credit.just_open_command_credited.float().mean()
        )
        log["/M10/release_ready_fraction"] = release_credit.release_ready.float().mean()
        log["/M10/gripper_openness_at_release_ready"] = (
            release_credit.openness * release_credit.release_ready.float()
        ).mean()
        log["/M10/just_released_fraction"] = release_credit.just_released.float().mean()
        log["/M10/physical_grasp_fraction"] = m4.physical_grasp.float().mean()
        log["/M10/stable_grasp_fraction"] = stable.stable_grasp.float().mean()
        log["/M10/lifted_fraction"] = lifted.float().mean()
        log["/M10/red_on_blue_fraction"] = placement.geometry_ok.float().mean()
        log["/M10/strict_candidate_fraction"] = success.candidate.float().mean()
        for stage_id, name in enumerate(("REACH", "GRASP", "LIFT", "TRANSPORT", "PLACE")):
            log[f"/M10/stage_{name.lower()}_fraction"] = (stage.stage == stage_id).float().mean()

    def __call__(self, env, m10_params: dict[str, int | float]) -> torch.Tensor:
        # ``m10_params`` is intentionally explicit: Isaac Lab 3.0 statically
        # matches RewardTermCfg.params keys to callable-class __call__
        # arguments.  Runtime values were validated and frozen into immutable
        # project configs in __init__; do not rebuild temporal state here.
        if not isinstance(m10_params, dict):
            raise TypeError("m10_params must be a dict")
        # A normal ManagerBasedRLEnv reset invokes RewardManager.reset before
        # training starts.  Keep a defensive lazy path for unusual custom use.
        if not bool(self.reward_tracker.initialized.all().item()):
            missing = (~self.reward_tracker.initialized).nonzero(as_tuple=False).flatten()
            self.reset(missing)

        state = read_grasp_state(env)
        m4 = physical_grasp_diagnostics(
            state.red_pos_w,
            state.left_tip_w,
            state.right_tip_w,
            state.finger_a_force_n,
            state.finger_b_force_n,
            cfg=self.physical_cfg,
        )
        stable = self.stable_tracker.update(m4.physical_grasp)

        new_reference = stable.just_became_stable & ~self._has_reference_red_z
        if bool(new_reference.any().item()):
            self._reference_red_z = torch.where(
                new_reference,
                state.red_pos_w[:, 2].to(self._reference_red_z.dtype),
                self._reference_red_z,
            )
            self._has_reference_red_z |= new_reference

        reference = torch.where(
            self._has_reference_red_z,
            self._reference_red_z,
            state.red_pos_w[:, 2].to(self._reference_red_z.dtype),
        )
        lift = lift_diagnostics(
            state.red_pos_w[:, 2],
            reference,
            stable.stable_grasp,
            cfg=self.lift_cfg,
        )
        lifted_now = lift.lifted & self._has_reference_red_z
        history = self.history.update(stable.stable_grasp, lifted_now)

        pstate = read_placement_state(
            env, gripper_open_tolerance_m=self.gripper_open_tolerance_m
        )
        goal = pstate.blue_pos_w.clone()
        goal[:, 2] = pstate.blue_pos_w[:, 2] + float(self.placement_cfg.target_height_diff_m)
        placement = red_on_blue_diagnostics(
            pstate.red_pos_w,
            pstate.blue_pos_w,
            pstate.red_lin_vel_w,
            pstate.red_ang_vel_w,
            pstate.blue_lin_vel_w,
            pstate.blue_ang_vel_w,
            cfg=self.placement_cfg,
        )
        success = self.success_tracker.update(
            placement,
            pstate.gripper_open,
            m4.physical_grasp,
            history.has_stable_grasped,
            history.has_lifted,
        )

        reach_distance = torch.linalg.vector_norm(state.ee_pos_w - state.red_pos_w, dim=-1)
        goal_distance = torch.linalg.vector_norm(state.red_pos_w - goal, dim=-1)
        stage = self.stage_tracker.update(
            reach_distance,
            stable.stable_grasp,
            lifted_now,
            goal_distance,
        )

        # M10-4: add release credit only after the existing stage machine has
        # reached PLACE and strict red-on-blue geometry/history are already
        # valid.  This does not alter stage transitions or the M8 potential.
        if not hasattr(env.cfg, "gripper_open_val"):
            raise RuntimeError("Environment cfg has no gripper_open_val.")
        closedness = gripper_closedness(
            pstate.gripper_joint_pos,
            open_joint_pos_m=float(env.cfg.gripper_open_val),
            closed_joint_pos_m=self.release_closed_joint_pos_m,
        )
        openness = 1.0 - closedness
        history_ok = history.has_stable_grasped & history.has_lifted
        release_ready = (
            (stage.stage == int(ManipulationStage.PLACE))
            & placement.geometry_ok
            & history_ok
        )
        released_now = release_ready & pstate.gripper_open & ~m4.physical_grasp
        if self.use_policy_gripper_token:
            policy_gripper_token, pre_action_release_ready = (
                self._consume_policy_gripper_context()
            )
            open_command = policy_gripper_token == M9B_OPEN_TOKEN
            open_command_ready = pre_action_release_ready
        else:
            # Continuous PPO: no discrete grip token. Use the actual gripper
            # state -- the command was "open" iff the gripper is currently open,
            # and release-ready is the state-based gate.
            open_command = pstate.gripper_open
            open_command_ready = release_ready
        release_credit = self.release_credit_tracker.update(
            openness,
            release_ready,
            released_now,
            open_command=open_command,
            open_command_ready=open_command_ready,
        )
        settle_shaping = self.settle_shaping_tracker.update(
            release_ready,
            placement.red_linear_speed_mps,
            placement.red_angular_speed_radps,
            placement.blue_linear_speed_mps,
            placement.blue_angular_speed_radps,
        )

        grasp_error = self._grasp_error(m4)
        target_lift_z = self._reference_red_z + float(self.lift_cfg.minimum_object_lift_delta_m)
        lift_residual = torch.abs(state.red_pos_w[:, 2].to(target_lift_z.dtype) - target_lift_z)
        lift_residual = torch.where(
            self._has_reference_red_z,
            lift_residual,
            torch.full_like(lift_residual, self.lift_cfg.minimum_object_lift_delta_m),
        )
        phi = stage_potential(
            stage.stage,
            StagePotentialInputs(
                reach_distance_m=reach_distance,
                grasp_error_m=grasp_error,
                lift_residual_m=lift_residual,
                transport_distance_m=goal_distance,
                place_distance_m=goal_distance,
                transport_scale_m=self._transport_scale_m,
            ),
            cfg=self.potential_cfg,
        )
        reward = self.reward_tracker.update(phi, stage.transition_count, success.just_succeeded)
        total_reward = reward.total_reward + release_credit.total_reward + settle_shaping.shaping_reward
        if not torch.isfinite(total_reward).all():
            raise RuntimeError("M10 stage-aware + release (+ optional diagnostic settle term) reward produced NaN/Inf")

        self.call_count += 1
        self.last_physical_grasp = m4.physical_grasp.clone()
        self.last_stable_grasp = stable.stable_grasp.clone()
        self.last_lifted = lifted_now.clone()
        self.last_geometry_ok = placement.geometry_ok.clone()
        self.last_settled = placement.settled.clone()
        self.last_success_candidate = success.candidate.clone()
        self.last_total_reward = total_reward.clone()
        self.last_shaping_reward = reward.shaping_reward.clone()
        self.last_transition_bonus = reward.transition_bonus.clone()
        self.last_success_bonus = reward.sparse_success_reward.clone()
        self.last_release_credit = release_credit.total_reward.clone()
        self.last_open_token_credit = release_credit.command_reward.clone()
        self.last_release_ready = release_credit.release_ready.clone()
        self.last_open_command_ready = release_credit.open_command_ready.clone()
        self.last_open_command = release_credit.open_command.clone()
        self.last_open_command_credited = release_credit.just_open_command_credited.clone()
        self.last_just_released = release_credit.just_released.clone()
        self.last_settle_potential = settle_shaping.potential.clone()
        self.last_settle_shaping_reward = settle_shaping.shaping_reward.clone()
        self.last_settle_speed_ratio = settle_shaping.normalized_speed_ratio.clone()
        self._write_step_logs(
            env, reward, release_credit, settle_shaping, stage, m4, stable, lifted_now, placement, success
        )
        return total_reward
