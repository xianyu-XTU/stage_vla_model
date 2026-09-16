"""Read-only strict M7 success observer for checkpoint evaluation.

The observer reuses the already-validated M4/M5/M6/M7 semantics while a
learned policy acts in Isaac Lab.  It does not alter reward, actions, or
terminations.  Evaluation scripts should prevent automatic reset during the
fixed evaluation window; otherwise Isaac Lab can reset before post-step state
is inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stage_vla.stages import (
    LiftConfig,
    PhysicalGraspConfig,
    RedOnBlueConfig,
    RedOnBlueSuccessConfig,
    RedOnBlueSuccessTracker,
    StableGraspConfig,
    StableGraspTracker,
    TaskHistoryTracker,
    lift_diagnostics,
    physical_grasp_diagnostics,
    red_on_blue_diagnostics,
)

from .state_readers import read_grasp_state, read_placement_state


@dataclass(frozen=True)
class StrictSuccessSummary:
    num_envs: int
    ever_physical_grasp: int
    ever_stable_grasp: int
    ever_lifted: int
    ever_geometry_ok: int
    ever_postlift_gripper_open: int
    ever_postlift_released: int
    ever_geometry_history_ok: int
    ever_geometry_gripper_open: int
    ever_geometry_not_physical: int
    ever_geometry_released: int
    ever_geometry_settled: int
    ever_geometry_released_history: int
    ever_geometry_settled_history: int
    ever_success_candidate: int
    max_geometry_streak_any_env: int
    envs_geometry_streak_ge_required: int
    strict_successes: int
    strict_success_rate: float


@dataclass(frozen=True)
class StrictBottleneckMasks:
    postlift_gripper_open: torch.Tensor
    postlift_released: torch.Tensor
    geometry_history_ok: torch.Tensor
    geometry_gripper_open: torch.Tensor
    geometry_not_physical: torch.Tensor
    geometry_released: torch.Tensor
    geometry_settled: torch.Tensor
    geometry_released_history: torch.Tensor
    geometry_settled_history: torch.Tensor
    candidate: torch.Tensor


def strict_bottleneck_masks(
    *,
    geometry_ok: torch.Tensor,
    settled: torch.Tensor,
    gripper_open: torch.Tensor,
    physical_grasp: torch.Tensor,
    has_stable_grasped: torch.Tensor,
    has_lifted: torch.Tensor,
) -> StrictBottleneckMasks:
    """Decompose the strict M7 terminal gate into simultaneous bottleneck masks.

    The helper is intentionally pure and does not alter tracker state.  It is
    used only by checkpoint evaluation so a 0% success result can be localized
    to release, settling, geometry persistence, or timing/coupling.
    """
    values = {
        "geometry_ok": torch.as_tensor(geometry_ok),
        "settled": torch.as_tensor(settled),
        "gripper_open": torch.as_tensor(gripper_open),
        "physical_grasp": torch.as_tensor(physical_grasp),
        "has_stable_grasped": torch.as_tensor(has_stable_grasped),
        "has_lifted": torch.as_tensor(has_lifted),
    }
    shape = values["geometry_ok"].shape
    for name, value in values.items():
        if value.dtype is not torch.bool:
            raise TypeError(f"{name} must be torch.bool")
        if value.shape != shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(shape)}")

    geometry = values["geometry_ok"]
    settled_mask = values["settled"]
    open_mask = values["gripper_open"]
    physical = values["physical_grasp"]
    history_ok = values["has_stable_grasped"] & values["has_lifted"]
    released = open_mask & ~physical
    postlift = values["has_lifted"]

    return StrictBottleneckMasks(
        postlift_gripper_open=postlift & open_mask,
        postlift_released=postlift & released,
        geometry_history_ok=geometry & history_ok,
        geometry_gripper_open=geometry & open_mask,
        geometry_not_physical=geometry & ~physical,
        geometry_released=geometry & released,
        geometry_settled=geometry & settled_mask,
        geometry_released_history=geometry & released & history_ok,
        geometry_settled_history=geometry & settled_mask & history_ok,
        candidate=geometry & settled_mask & released & history_ok,
    )


class StrictM7SuccessObserver:
    """Vectorized one-episode strict success observer."""

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
    ) -> None:
        self.base_env = base_env
        self.num_envs = int(base_env.num_envs)
        self.device = torch.device(base_env.device)
        self.physical_cfg = physical_cfg
        self.lift_cfg = lift_cfg
        self.placement_cfg = placement_cfg
        self.gripper_open_tolerance_m = float(gripper_open_tolerance_m)

        self.stable = StableGraspTracker(
            num_envs=self.num_envs, device=self.device, cfg=stable_cfg
        )
        self.history = TaskHistoryTracker(num_envs=self.num_envs, device=self.device)
        self.success = RedOnBlueSuccessTracker(
            num_envs=self.num_envs, device=self.device, cfg=success_cfg
        )

        self._reference_red_z = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._has_reference = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._ever_physical = torch.zeros_like(self._has_reference)
        self._ever_stable = torch.zeros_like(self._has_reference)
        self._ever_lifted = torch.zeros_like(self._has_reference)
        self._ever_geometry = torch.zeros_like(self._has_reference)
        self._ever_postlift_gripper_open = torch.zeros_like(self._has_reference)
        self._ever_postlift_released = torch.zeros_like(self._has_reference)
        self._ever_geometry_history_ok = torch.zeros_like(self._has_reference)
        self._ever_geometry_gripper_open = torch.zeros_like(self._has_reference)
        self._ever_geometry_not_physical = torch.zeros_like(self._has_reference)
        self._ever_geometry_released = torch.zeros_like(self._has_reference)
        self._ever_geometry_settled = torch.zeros_like(self._has_reference)
        self._ever_geometry_released_history = torch.zeros_like(self._has_reference)
        self._ever_geometry_settled_history = torch.zeros_like(self._has_reference)
        self._ever_candidate = torch.zeros_like(self._has_reference)
        self._geometry_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_geometry_streak = torch.zeros_like(self._geometry_streak)
        self._ever_success = torch.zeros_like(self._has_reference)
        self._first_success_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

    @property
    def ever_success_mask(self) -> torch.Tensor:
        return self._ever_success.clone()

    @property
    def first_success_step(self) -> torch.Tensor:
        return self._first_success_step.clone()

    def update(self, *, step_index: int) -> None:
        gstate = read_grasp_state(self.base_env)
        m4 = physical_grasp_diagnostics(
            gstate.red_pos_w,
            gstate.left_tip_w,
            gstate.right_tip_w,
            gstate.finger_a_force_n,
            gstate.finger_b_force_n,
            cfg=self.physical_cfg,
        )
        stable = self.stable.update(m4.physical_grasp)

        new_reference = stable.just_became_stable & ~self._has_reference
        self._reference_red_z = torch.where(
            new_reference,
            gstate.red_pos_w[:, 2].to(self._reference_red_z.dtype),
            self._reference_red_z,
        )
        self._has_reference |= new_reference

        reference = torch.where(
            self._has_reference,
            self._reference_red_z,
            gstate.red_pos_w[:, 2].to(self._reference_red_z.dtype),
        )
        lift = lift_diagnostics(
            gstate.red_pos_w[:, 2],
            reference,
            stable.stable_grasp,
            cfg=self.lift_cfg,
        )
        lifted_now = lift.lifted & self._has_reference
        history = self.history.update(stable.stable_grasp, lifted_now)

        pstate = read_placement_state(
            self.base_env,
            gripper_open_tolerance_m=self.gripper_open_tolerance_m,
        )
        placement = red_on_blue_diagnostics(
            pstate.red_pos_w,
            pstate.blue_pos_w,
            pstate.red_lin_vel_w,
            pstate.red_ang_vel_w,
            pstate.blue_lin_vel_w,
            pstate.blue_ang_vel_w,
            cfg=self.placement_cfg,
        )
        success = self.success.update(
            placement,
            pstate.gripper_open,
            m4.physical_grasp,
            history.has_stable_grasped,
            history.has_lifted,
        )

        bottlenecks = strict_bottleneck_masks(
            geometry_ok=placement.geometry_ok,
            settled=placement.settled,
            gripper_open=pstate.gripper_open,
            physical_grasp=m4.physical_grasp,
            has_stable_grasped=history.has_stable_grasped,
            has_lifted=history.has_lifted,
        )
        if not torch.equal(bottlenecks.candidate, success.candidate):
            raise RuntimeError("strict bottleneck decomposition disagrees with M7 candidate truth")

        self._ever_physical |= m4.physical_grasp
        self._ever_stable |= history.has_stable_grasped
        self._ever_lifted |= history.has_lifted
        self._ever_geometry |= placement.geometry_ok
        self._ever_postlift_gripper_open |= bottlenecks.postlift_gripper_open
        self._ever_postlift_released |= bottlenecks.postlift_released
        self._ever_geometry_history_ok |= bottlenecks.geometry_history_ok
        self._ever_geometry_gripper_open |= bottlenecks.geometry_gripper_open
        self._ever_geometry_not_physical |= bottlenecks.geometry_not_physical
        self._ever_geometry_released |= bottlenecks.geometry_released
        self._ever_geometry_settled |= bottlenecks.geometry_settled
        self._ever_geometry_released_history |= bottlenecks.geometry_released_history
        self._ever_geometry_settled_history |= bottlenecks.geometry_settled_history
        self._ever_candidate |= success.candidate

        self._geometry_streak = torch.where(
            placement.geometry_ok,
            self._geometry_streak + 1,
            torch.zeros_like(self._geometry_streak),
        )
        self._max_geometry_streak = torch.maximum(
            self._max_geometry_streak, self._geometry_streak
        )

        first_now = success.just_succeeded & ~self._ever_success
        self._first_success_step = torch.where(
            first_now,
            torch.full_like(self._first_success_step, int(step_index)),
            self._first_success_step,
        )
        self._ever_success |= success.success

    def summary(self) -> StrictSuccessSummary:
        success_count = int(self._ever_success.sum().item())
        required = int(self.success.cfg.required_settle_steps)
        return StrictSuccessSummary(
            num_envs=self.num_envs,
            ever_physical_grasp=int(self._ever_physical.sum().item()),
            ever_stable_grasp=int(self._ever_stable.sum().item()),
            ever_lifted=int(self._ever_lifted.sum().item()),
            ever_geometry_ok=int(self._ever_geometry.sum().item()),
            ever_postlift_gripper_open=int(self._ever_postlift_gripper_open.sum().item()),
            ever_postlift_released=int(self._ever_postlift_released.sum().item()),
            ever_geometry_history_ok=int(self._ever_geometry_history_ok.sum().item()),
            ever_geometry_gripper_open=int(self._ever_geometry_gripper_open.sum().item()),
            ever_geometry_not_physical=int(self._ever_geometry_not_physical.sum().item()),
            ever_geometry_released=int(self._ever_geometry_released.sum().item()),
            ever_geometry_settled=int(self._ever_geometry_settled.sum().item()),
            ever_geometry_released_history=int(self._ever_geometry_released_history.sum().item()),
            ever_geometry_settled_history=int(self._ever_geometry_settled_history.sum().item()),
            ever_success_candidate=int(self._ever_candidate.sum().item()),
            max_geometry_streak_any_env=int(self._max_geometry_streak.max().item()),
            envs_geometry_streak_ge_required=int((self._max_geometry_streak >= required).sum().item()),
            strict_successes=success_count,
            strict_success_rate=success_count / float(self.num_envs),
        )
