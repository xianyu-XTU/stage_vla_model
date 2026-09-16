"""M7 red-on-blue success composition.

A success candidate requires all of:

- red-on-blue geometry;
- both red and blue are low-speed / settled;
- physical gripper is open;
- M4 physical_grasp is false (the red cube has actually been released);
- episode history proves a genuine stable grasp happened earlier;
- episode history proves a genuine lift happened earlier.

The candidate must remain true for N consecutive environment steps before
``success=True``. This avoids a one-frame contact/velocity coincidence.

No reward is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from .placement import RedOnBlueDiagnostics


@dataclass(frozen=True)
class RedOnBlueSuccessConfig:
    required_settle_steps: int

    def validate(self) -> None:
        if not isinstance(self.required_settle_steps, int):
            raise TypeError("required_settle_steps must be int")
        if self.required_settle_steps < 1:
            raise ValueError("required_settle_steps must be >= 1")


@dataclass(frozen=True)
class RedOnBlueSuccessDiagnostics:
    success: Tensor
    just_succeeded: Tensor
    candidate: Tensor
    released: Tensor
    history_ok: Tensor
    candidate_streak_steps: Tensor


def success_candidate(
    placement: RedOnBlueDiagnostics,
    gripper_open: Tensor,
    physical_grasp: Tensor,
    has_stable_grasped: Tensor,
    has_lifted: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose the current-frame non-temporal M7 gates."""
    geometry = torch.as_tensor(placement.geometry_ok)
    settled = torch.as_tensor(placement.settled, device=geometry.device)
    open_mask = torch.as_tensor(gripper_open, device=geometry.device)
    physical = torch.as_tensor(physical_grasp, device=geometry.device)
    history_stable = torch.as_tensor(
        has_stable_grasped, device=geometry.device
    )
    history_lifted = torch.as_tensor(has_lifted, device=geometry.device)

    expected_shape = geometry.shape
    for name, value in (
        ("placement.settled", settled),
        ("gripper_open", open_mask),
        ("physical_grasp", physical),
        ("has_stable_grasped", history_stable),
        ("has_lifted", history_lifted),
    ):
        if value.dtype is not torch.bool:
            raise TypeError(f"{name} must be torch.bool")
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != "
                f"{tuple(expected_shape)}"
            )

    released = open_mask & ~physical
    history_ok = history_stable & history_lifted
    candidate = geometry & settled & released & history_ok
    return candidate, released, history_ok


class RedOnBlueSuccessTracker:
    """Vectorized consecutive-candidate tracker for final M7 success."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: RedOnBlueSuccessConfig,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        cfg.validate()

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    @property
    def streak_steps(self) -> Tensor:
        return self._streak.clone()

    def reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        if env_ids is None:
            self._streak.zero_()
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
            self._streak[ids] = 0

    def update(
        self,
        placement: RedOnBlueDiagnostics,
        gripper_open: Tensor,
        physical_grasp: Tensor,
        has_stable_grasped: Tensor,
        has_lifted: Tensor,
    ) -> RedOnBlueSuccessDiagnostics:
        candidate, released, history_ok = success_candidate(
            placement,
            gripper_open,
            physical_grasp,
            has_stable_grasped,
            has_lifted,
        )
        candidate = candidate.to(self.device)

        if candidate.shape != (self.num_envs,):
            raise ValueError(
                f"candidate must have shape ({self.num_envs},), "
                f"got {tuple(candidate.shape)}"
            )

        self._streak = torch.where(
            candidate,
            self._streak + 1,
            torch.zeros_like(self._streak),
        )
        required = self.cfg.required_settle_steps
        success = self._streak >= required
        just = self._streak == required

        return RedOnBlueSuccessDiagnostics(
            success=success.clone(),
            just_succeeded=just.clone(),
            candidate=candidate.clone(),
            released=released.to(self.device).clone(),
            history_ok=history_ok.to(self.device).clone(),
            candidate_streak_steps=self._streak.clone(),
        )
