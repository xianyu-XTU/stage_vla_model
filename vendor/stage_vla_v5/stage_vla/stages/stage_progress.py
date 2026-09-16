"""M8 monotonic manipulation-stage tracker.

The project uses five explicit policy/reward stages:

    REACH -> GRASP -> LIFT -> TRANSPORT -> PLACE

This is a project-specific extension of StARe's event-driven stage separation:
StARe's pick-place example uses Reach -> Grasp -> Transport -> Place, while
its generic stage library also defines Lift.  Because this project already has
an independently verified M6 lift event, Lift is kept as an explicit stage.

The tracker is deliberately monotonic within one episode.  Losing a current
physical grasp later does not silently rewind the stage; failure/hold evidence
belongs in reward/success gates, while episode reset explicitly resets stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class ManipulationStage(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT = 2
    TRANSPORT = 3
    PLACE = 4


STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in ManipulationStage)


@dataclass(frozen=True)
class StageProgressConfig:
    """Geometric transition thresholds for the project stage separator."""

    reach_transition_distance_m: float
    place_transition_distance_m: float

    def validate(self) -> None:
        if self.reach_transition_distance_m <= 0:
            raise ValueError("reach_transition_distance_m must be > 0")
        if self.place_transition_distance_m <= 0:
            raise ValueError("place_transition_distance_m must be > 0")


@dataclass(frozen=True)
class StageProgressDiagnostics:
    previous_stage: Tensor
    stage: Tensor
    just_transitioned: Tensor
    transition_count: Tensor


def stage_name(stage: int | ManipulationStage) -> str:
    return ManipulationStage(int(stage)).name


class StageProgressTracker:
    """Vectorized, causal, monotonic stage state with selective reset."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: StageProgressConfig,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._stage = torch.full(
            (self.num_envs,),
            int(ManipulationStage.REACH),
            dtype=torch.long,
            device=self.device,
        )

    @property
    def stage(self) -> Tensor:
        return self._stage.clone()

    def reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        if env_ids is None:
            self._stage.fill_(int(ManipulationStage.REACH))
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be 1-D")
        if ids.numel() and (
            torch.any(ids < 0) or torch.any(ids >= self.num_envs)
        ):
            raise IndexError("env_ids out of range")
        if ids.numel():
            self._stage[ids] = int(ManipulationStage.REACH)

    def _mask(self, name: str, value: Tensor, *, dtype=None) -> Tensor:
        out = torch.as_tensor(value, device=self.device)
        if out.shape != (self.num_envs,):
            raise ValueError(
                f"{name} must have shape ({self.num_envs},), got {tuple(out.shape)}"
            )
        if dtype is not None and out.dtype is not dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {out.dtype}")
        return out

    def update(
        self,
        reach_distance_m: Tensor,
        stable_grasp: Tensor,
        lifted: Tensor,
        object_goal_distance_m: Tensor,
    ) -> StageProgressDiagnostics:
        """Advance stages from current task evidence.

        Multiple causal boundaries may be crossed in one environment step if
        the caller supplies evidence that already satisfies them.  The returned
        ``transition_count`` records how many stage boundaries were crossed.
        """
        reach = self._mask("reach_distance_m", reach_distance_m)
        goal = self._mask("object_goal_distance_m", object_goal_distance_m)
        stable = self._mask("stable_grasp", stable_grasp, dtype=torch.bool)
        lifted_now = self._mask("lifted", lifted, dtype=torch.bool)

        if not torch.isfinite(reach).all() or not torch.isfinite(goal).all():
            raise ValueError("stage distances contain NaN/Inf")
        if torch.any(reach < 0) or torch.any(goal < 0):
            raise ValueError("stage distances must be non-negative")

        previous = self._stage.clone()
        stage = self._stage.clone()

        # Reach -> Grasp: geometry is close enough for fine grasp alignment.
        stage = torch.where(
            (stage == int(ManipulationStage.REACH))
            & (reach <= self.cfg.reach_transition_distance_m),
            torch.full_like(stage, int(ManipulationStage.GRASP)),
            stage,
        )

        # Grasp -> Lift: use the verified M5 temporal truth, not proximity alone.
        stage = torch.where(
            (stage == int(ManipulationStage.GRASP)) & stable,
            torch.full_like(stage, int(ManipulationStage.LIFT)),
            stage,
        )

        # Lift -> Transport: use the verified M6 actual-object lift truth.
        stage = torch.where(
            (stage == int(ManipulationStage.LIFT)) & lifted_now,
            torch.full_like(stage, int(ManipulationStage.TRANSPORT)),
            stage,
        )

        # Transport -> Place: near-goal geometry starts fine placement.
        stage = torch.where(
            (stage == int(ManipulationStage.TRANSPORT))
            & (goal <= self.cfg.place_transition_distance_m),
            torch.full_like(stage, int(ManipulationStage.PLACE)),
            stage,
        )

        self._stage = stage
        transition_count = stage - previous
        if torch.any(transition_count < 0):
            raise RuntimeError("stage tracker regressed unexpectedly")

        return StageProgressDiagnostics(
            previous_stage=previous,
            stage=stage.clone(),
            just_transitioned=transition_count > 0,
            transition_count=transition_count.clone(),
        )
