"""M5 pure-PyTorch temporal stable-grasp tracker.

M4 answers:
    "Is the current frame a physical grasp?"

M5 answers:
    "Has physical_grasp remained true for N consecutive environment steps?"

The tracker is deliberately explicit state:
- one counter per vectorized environment;
- false immediately resets that environment's counter to zero;
- ``stable_grasp`` is true only while the current streak is >= N;
- it is NOT a permanent history latch;
- reset(env_ids) supports selective episode resets.

A permanent "has ever grasped" latch belongs to a later task-state/success layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class StableGraspConfig:
    required_consecutive_steps: int

    def validate(self) -> None:
        if not isinstance(self.required_consecutive_steps, int):
            raise TypeError("required_consecutive_steps must be an int")
        if self.required_consecutive_steps < 1:
            raise ValueError("required_consecutive_steps must be >= 1")


@dataclass(frozen=True)
class StableGraspDiagnostics:
    stable_grasp: Tensor
    just_became_stable: Tensor
    streak_steps: Tensor


class StableGraspTracker:
    """Vectorized consecutive-step tracker with explicit reset semantics."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: StableGraspConfig,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        cfg.validate()

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._streak_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

    @property
    def streak_steps(self) -> Tensor:
        """Current consecutive-physical-grasp count (read-only clone)."""
        return self._streak_steps.clone()

    def reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        """Reset all environments or a selected subset."""
        if env_ids is None:
            self._streak_steps.zero_()
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be a 1-D sequence/tensor of indices")
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError(
                f"env_ids out of range for num_envs={self.num_envs}: {ids.tolist()}"
            )
        self._streak_steps[ids] = 0

    def update(self, physical_grasp: Tensor) -> StableGraspDiagnostics:
        """Consume exactly one environment-step physical_grasp mask."""
        mask = torch.as_tensor(physical_grasp, device=self.device)
        if mask.dtype is not torch.bool:
            raise TypeError(
                f"physical_grasp must have dtype torch.bool, got {mask.dtype}"
            )
        if mask.shape != (self.num_envs,):
            raise ValueError(
                f"physical_grasp must have shape ({self.num_envs},), got {tuple(mask.shape)}"
            )

        required = self.cfg.required_consecutive_steps

        # Increment only true envs; any false frame immediately breaks the streak.
        self._streak_steps = torch.where(
            mask,
            self._streak_steps + 1,
            torch.zeros_like(self._streak_steps),
        )

        stable = self._streak_steps >= required
        just_became = self._streak_steps == required

        return StableGraspDiagnostics(
            stable_grasp=stable.clone(),
            just_became_stable=just_became.clone(),
            streak_steps=self._streak_steps.clone(),
        )
