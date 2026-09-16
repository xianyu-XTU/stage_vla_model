"""Runtime M8 observer for validating stage-aware reward on the M7 trajectory.

This observer does *not* replace the environment reward yet.  It watches every
post-physics state of the already-verified scripted M7-R5 trajectory and feeds
M4/M5/M6/M7 evidence into the pure-PyTorch stage separator and reward tracker.

M9 will decide how the validated reward is exposed to PPO.  Keeping M8 as an
observer avoids changing the task reward before its semantics have been checked
end-to-end in Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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
    stage_name,
    stage_potential,
)

from .state_readers import read_grasp_state, read_placement_state


@dataclass(frozen=True)
class M8RewardSummary:
    total_reward_sum: float
    shaping_reward_sum: float
    transition_bonus_sum: float
    success_bonus_sum: float
    min_step_reward: float
    max_step_reward: float
    nonzero_shaping_steps: int
    total_stage_transitions: int
    success_events: int
    stage_step_counts: tuple[int, int, int, int, int]
    visited_stage_ids: tuple[int, ...]


class StageRewardObserver:
    """Single-call-per-step M8 reward observer for a vectorized Isaac env."""

    def __init__(
        self,
        base_env,
        *,
        physical_cfg: PhysicalGraspConfig,
        stable_cfg: StableGraspConfig,
        lift_cfg: LiftConfig,
        placement_cfg: RedOnBlueConfig,
        success_cfg: RedOnBlueSuccessConfig,
        gripper_open_tolerance_m: float,
        cube_size_m: float,
        reward_cfg: StageAwareRewardConfig,
    ) -> None:
        if cube_size_m <= 0:
            raise ValueError("cube_size_m must be > 0")
        if gripper_open_tolerance_m < 0:
            raise ValueError("gripper_open_tolerance_m must be >= 0")

        self.base_env = base_env
        self.num_envs = int(base_env.num_envs)
        self.device = torch.device(base_env.device)
        self.physical_cfg = physical_cfg
        self.lift_cfg = lift_cfg
        self.placement_cfg = placement_cfg
        self.gripper_open_tolerance_m = gripper_open_tolerance_m
        self.cube_size_m = float(cube_size_m)

        self.stable_tracker = StableGraspTracker(
            num_envs=self.num_envs,
            device=self.device,
            cfg=stable_cfg,
        )
        self.history = TaskHistoryTracker(
            num_envs=self.num_envs,
            device=self.device,
        )
        self.success_tracker = RedOnBlueSuccessTracker(
            num_envs=self.num_envs,
            device=self.device,
            cfg=success_cfg,
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
            lift_scale_m=lift_cfg.minimum_object_lift_delta_m,
        )
        self.reward_tracker = StageAwareRewardTracker(
            num_envs=self.num_envs,
            device=self.device,
            cfg=reward_cfg,
        )

        self._reference_red_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._has_reference_red_z = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._transport_scale_m = torch.full(
            (self.num_envs,), self.cube_size_m, dtype=torch.float32, device=self.device
        )

        self._stage_counts = torch.zeros(5, dtype=torch.long)
        self._visited: set[int] = set()
        self._transition_total = 0
        self._success_events = 0
        self._nonzero_shaping_steps = 0
        self._total_reward_sum = 0.0
        self._shaping_reward_sum = 0.0
        self._transition_bonus_sum = 0.0
        self._success_bonus_sum = 0.0
        self._min_reward = float("inf")
        self._max_reward = float("-inf")
        self._initialized = False

    def _goal_position(self, red_dtype: torch.dtype):
        pstate = read_placement_state(
            self.base_env,
            gripper_open_tolerance_m=self.gripper_open_tolerance_m,
        )
        offset = torch.zeros_like(pstate.blue_pos_w)
        offset[:, 2] = float(self.placement_cfg.target_height_diff_m)
        return pstate, pstate.blue_pos_w + offset

    def _m4(self):
        state = read_grasp_state(self.base_env)
        diag = physical_grasp_diagnostics(
            state.red_pos_w,
            state.left_tip_w,
            state.right_tip_w,
            state.finger_a_force_n,
            state.finger_b_force_n,
            cfg=self.physical_cfg,
        )
        return diag, state

    @staticmethod
    def _grasp_error(m4) -> torch.Tensor:
        mean_height_error = 0.5 * (
            m4.left_height_error_m + m4.right_height_error_m
        )
        return torch.sqrt(m4.radial_error_m.square() + mean_height_error.square())

    def initialize_from_current_state(self) -> None:
        """Reset M8 state and seed Phi(s_0) without emitting a reward."""
        self.stable_tracker.reset()
        self.history.reset()
        self.success_tracker.reset()
        self.stage_tracker.reset()
        self._has_reference_red_z.zero_()
        self._reference_red_z.zero_()

        self._stage_counts.zero_()
        self._visited.clear()
        self._transition_total = 0
        self._success_events = 0
        self._nonzero_shaping_steps = 0
        self._total_reward_sum = 0.0
        self._shaping_reward_sum = 0.0
        self._transition_bonus_sum = 0.0
        self._success_bonus_sum = 0.0
        self._min_reward = float("inf")
        self._max_reward = float("-inf")

        m4, state = self._m4()
        _, goal = self._goal_position(state.red_pos_w.dtype)
        goal_distance = torch.linalg.vector_norm(state.red_pos_w - goal, dim=-1)
        # StARe transport normalization uses task displacement.  The object-size
        # floor prevents a degenerate near-zero reset from dividing by zero.
        self._transport_scale_m = torch.maximum(
            goal_distance.to(torch.float32),
            torch.full_like(goal_distance.to(torch.float32), self.cube_size_m),
        )

        reach_distance = torch.linalg.vector_norm(
            state.ee_pos_w - state.red_pos_w, dim=-1
        )
        grasp_error = self._grasp_error(m4)
        lift_residual = torch.full_like(
            reach_distance, float(self.lift_cfg.minimum_object_lift_delta_m)
        )
        stage = self.stage_tracker.stage
        phi = stage_potential(
            stage,
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
        self.reward_tracker.reset(phi)
        self._visited.add(int(ManipulationStage.REACH))
        self._initialized = True

        print("\n=== M8 STAGE-AWARE REWARD OBSERVER ===")
        print("stages                     : REACH -> GRASP -> LIFT -> TRANSPORT -> PLACE")
        print(f"transport scale (env0)     : {float(self._transport_scale_m[0]):.5f} m")
        print(f"initial potential (env0)   : {float(phi[0]):.6f}")

    def after_step(self, *, global_step: int, phase: str) -> None:
        if not self._initialized:
            raise RuntimeError("StageRewardObserver must be initialized after env.reset()")

        m4, state = self._m4()
        stable = self.stable_tracker.update(m4.physical_grasp)

        new_reference = stable.just_became_stable & ~self._has_reference_red_z
        if bool(new_reference.any().item()):
            self._reference_red_z = torch.where(
                new_reference,
                state.red_pos_w[:, 2].to(self._reference_red_z.dtype),
                self._reference_red_z,
            )
            self._has_reference_red_z |= new_reference

        lifted_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if bool(self._has_reference_red_z.any().item()):
            # For environments without a reference, use current z as a harmless
            # placeholder and mask the result away below.
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

        pstate, goal = self._goal_position(state.red_pos_w.dtype)
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

        reach_distance = torch.linalg.vector_norm(
            state.ee_pos_w - state.red_pos_w, dim=-1
        )
        goal_distance = torch.linalg.vector_norm(state.red_pos_w - goal, dim=-1)
        stage = self.stage_tracker.update(
            reach_distance,
            stable.stable_grasp,
            lifted_now,
            goal_distance,
        )

        grasp_error = self._grasp_error(m4)
        if bool(self._has_reference_red_z.any().item()):
            target_lift_z = self._reference_red_z + float(
                self.lift_cfg.minimum_object_lift_delta_m
            )
            lift_residual = torch.abs(
                state.red_pos_w[:, 2].to(target_lift_z.dtype) - target_lift_z
            )
            lift_residual = torch.where(
                self._has_reference_red_z,
                lift_residual,
                torch.full_like(lift_residual, self.lift_cfg.minimum_object_lift_delta_m),
            )
        else:
            lift_residual = torch.full_like(
                reach_distance, float(self.lift_cfg.minimum_object_lift_delta_m)
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
        reward = self.reward_tracker.update(
            phi,
            stage.transition_count,
            success.just_succeeded,
        )

        if not torch.isfinite(reward.total_reward).all():
            raise RuntimeError("M8 reward produced NaN/Inf")

        for stage_id in stage.stage.detach().cpu().tolist():
            self._stage_counts[int(stage_id)] += 1
            self._visited.add(int(stage_id))
        transition_count = int(stage.transition_count.sum().item())
        self._transition_total += transition_count
        self._success_events += int(success.just_succeeded.sum().item())
        self._nonzero_shaping_steps += int(
            (torch.abs(reward.shaping_reward) > 1.0e-8).sum().item()
        )

        self._total_reward_sum += float(reward.total_reward.sum().item())
        self._shaping_reward_sum += float(reward.shaping_reward.sum().item())
        self._transition_bonus_sum += float(reward.transition_bonus.sum().item())
        self._success_bonus_sum += float(reward.sparse_success_reward.sum().item())
        self._min_reward = min(self._min_reward, float(reward.total_reward.min().item()))
        self._max_reward = max(self._max_reward, float(reward.total_reward.max().item()))

        if bool(stage.just_transitioned[0].item()):
            old = stage_name(int(stage.previous_stage[0].item()))
            new = stage_name(int(stage.stage[0].item()))
            print(
                f"[M8 STAGE @ global {global_step}] {old} -> {new} "
                f"(crossed={int(stage.transition_count[0].item())}, phase={phase}) | "
                f"phi={float(phi[0]):.5f} reward={float(reward.total_reward[0]):+.5f}"
            )
        if bool(success.just_succeeded[0].item()):
            print(
                f"[M8 SUCCESS REWARD @ global {global_step}] "
                f"sparse=+{float(reward.sparse_success_reward[0]):.3f} "
                f"total={float(reward.total_reward[0]):+.5f}"
            )

    def summary(self) -> M8RewardSummary:
        return M8RewardSummary(
            total_reward_sum=self._total_reward_sum,
            shaping_reward_sum=self._shaping_reward_sum,
            transition_bonus_sum=self._transition_bonus_sum,
            success_bonus_sum=self._success_bonus_sum,
            min_step_reward=0.0 if self._min_reward == float("inf") else self._min_reward,
            max_step_reward=0.0 if self._max_reward == float("-inf") else self._max_reward,
            nonzero_shaping_steps=self._nonzero_shaping_steps,
            total_stage_transitions=self._transition_total,
            success_events=self._success_events,
            stage_step_counts=tuple(int(x) for x in self._stage_counts.tolist()),
            visited_stage_ids=tuple(sorted(self._visited)),
        )

    def print_summary(self) -> M8RewardSummary:
        s = self.summary()
        print("\n=== M8 STAGE-AWARE REWARD RESULT ===")
        print(f"visited stages          : {[stage_name(x) for x in s.visited_stage_ids]}")
        print(f"stage step counts       : {dict(zip([x.name for x in ManipulationStage], s.stage_step_counts))}")
        print(f"stage transitions       : {s.total_stage_transitions}")
        print(f"success reward events   : {s.success_events}")
        print(f"nonzero shaping steps   : {s.nonzero_shaping_steps}")
        print(f"shaping reward sum      : {s.shaping_reward_sum:+.6f}")
        print(f"transition bonus sum    : {s.transition_bonus_sum:+.6f}")
        print(f"success bonus sum       : {s.success_bonus_sum:+.6f}")
        print(f"total reward sum        : {s.total_reward_sum:+.6f}")
        print(f"step reward min / max   : {s.min_step_reward:+.6f} / {s.max_step_reward:+.6f}")
        return s

    def assert_m8_pass(self) -> None:
        s = self.summary()
        expected = tuple(int(x) for x in ManipulationStage)
        if s.visited_stage_ids != expected:
            raise RuntimeError(
                f"M8 FAIL: expected all stages {expected}, visited {s.visited_stage_ids}"
            )
        if s.total_stage_transitions < 4:
            raise RuntimeError(
                f"M8 FAIL: expected at least four causal stage boundaries, got {s.total_stage_transitions}"
            )
        if s.success_events < 1:
            raise RuntimeError("M8 FAIL: no history-gated M7 success reward event observed")
        if s.nonzero_shaping_steps < 1:
            raise RuntimeError("M8 FAIL: dense potential shaping was identically zero")
