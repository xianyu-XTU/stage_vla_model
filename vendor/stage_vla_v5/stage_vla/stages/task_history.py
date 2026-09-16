"""Explicit episode-history latches required by M7 success.

The project must reject accidental "knock it up and it lands correctly" success.
M7 therefore remembers whether each environment has genuinely:

1. reached M5 stable_grasp;
2. reached M6 lifted after that stable grasp.

These are permanent *within the current episode* and must be reset explicitly.
They are intentionally separate from M5's non-latching stable_grasp streak.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TaskHistoryDiagnostics:
    has_stable_grasped: Tensor
    has_lifted: Tensor


class TaskHistoryTracker:
    def __init__(self, *, num_envs: int, device: str | torch.device) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._has_stable_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._has_lifted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def has_stable_grasped(self) -> Tensor:
        return self._has_stable_grasped.clone()

    @property
    def has_lifted(self) -> Tensor:
        return self._has_lifted.clone()

    def _ids(self, env_ids: Sequence[int] | Tensor | None) -> Tensor | None:
        if env_ids is None:
            return None
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be 1-D")
        if ids.numel() and (
            torch.any(ids < 0) or torch.any(ids >= self.num_envs)
        ):
            raise IndexError(
                f"env_ids out of range for num_envs={self.num_envs}: "
                f"{ids.tolist()}"
            )
        return ids

    def reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        ids = self._ids(env_ids)
        if ids is None:
            self._has_stable_grasped.zero_()
            self._has_lifted.zero_()
        elif ids.numel():
            self._has_stable_grasped[ids] = False
            self._has_lifted[ids] = False

    def update(
        self,
        stable_grasp: Tensor,
        lifted: Tensor,
    ) -> TaskHistoryDiagnostics:
        stable = torch.as_tensor(stable_grasp, device=self.device)
        lifted_now = torch.as_tensor(lifted, device=self.device)

        for name, value in (("stable_grasp", stable), ("lifted", lifted_now)):
            if value.dtype is not torch.bool:
                raise TypeError(f"{name} must be torch.bool")
            if value.shape != (self.num_envs,):
                raise ValueError(
                    f"{name} must have shape ({self.num_envs},), "
                    f"got {tuple(value.shape)}"
                )

        self._has_stable_grasped |= stable

        # Enforce causal order. Even if an upstream caller accidentally supplies
        # lifted=True in an environment that never had stable_grasp, do not latch it.
        self._has_lifted |= lifted_now & self._has_stable_grasped

        return TaskHistoryDiagnostics(
            has_stable_grasped=self._has_stable_grasped.clone(),
            has_lifted=self._has_lifted.clone(),
        )
