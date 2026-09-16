"""M10-12-F1-E bounded post-OPEN translation hold.

F1 proved that terminal GRIP likelihood credit can be localized without
destroying the real XYZ+GRIP co-action used to issue OPEN.  The remaining
bottleneck is post-release stability.  F1-E therefore preserves the complete
trigger action and suppresses only XYZ translation on a bounded number of
*future* policy steps after the first release-ready OPEN command.

The historical H4X ablation zeroed XYZ in every pre-action release-ready state,
including the OPEN trigger itself.  This tracker is deliberately different:

1. apply any already-armed hold to the current raw action;
2. execute/credit the current policy action normally;
3. if it is the episode's first ``release_ready(s_t) & OPEN`` action, arm a
   one-shot countdown for subsequent actions.

Rotation and GRIP are never changed.  The helper is pure torch so its temporal
semantics can be regression-tested without Isaac Lab.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

M10_F1E_POST_OPEN_HOLD_SEMANTICS = (
    "after_first_pre_action_release_ready_open_zero_xyz_on_bounded_future_steps"
)
M10_F1E_HOLD_ACTIVE_EXTRAS_KEY = "m10_f1e_post_open_xyz_hold_active"
M10_F1E_HOLD_TRIGGER_EXTRAS_KEY = "m10_f1e_post_open_xyz_hold_triggered"
M10_F1E_HOLD_REMAINING_EXTRAS_KEY = "m10_f1e_post_open_xyz_hold_remaining"


@dataclass(frozen=True)
class M10PostOpenHoldConfig:
    hold_steps: int

    def validate(self) -> None:
        if not isinstance(self.hold_steps, int):
            raise TypeError("hold_steps must be int")
        if self.hold_steps < 1:
            raise ValueError("hold_steps must be >= 1")


@dataclass(frozen=True)
class M10PostOpenHoldResult:
    raw_action: Tensor
    hold_mask: Tensor
    xyz_before: Tensor
    xyz_after: Tensor
    remaining_before: Tensor
    remaining_after: Tensor


class M10PostOpenXYZHoldTracker:
    """One-shot, per-environment post-OPEN XYZ hold countdown."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: M10PostOpenHoldConfig,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._triggered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def remaining_steps(self) -> Tensor:
        return self._remaining.clone()

    @property
    def triggered(self) -> Tensor:
        return self._triggered.clone()

    def _ids(self, env_ids: Sequence[int] | Tensor | slice | None) -> Tensor:
        all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be 1-D")
        if ids.numel() and bool(((ids < 0) | (ids >= self.num_envs)).any().item()):
            raise IndexError("env_ids out of range")
        return ids

    def reset(self, env_ids: Sequence[int] | Tensor | slice | None = None) -> None:
        ids = self._ids(env_ids)
        self._remaining[ids] = 0
        self._triggered[ids] = False

    def arm(self, release_ready: Tensor, open_command: Tensor) -> Tensor:
        """Arm future-step hold after the first release-ready OPEN in an episode."""
        ready = torch.as_tensor(release_ready, device=self.device)
        open_cmd = torch.as_tensor(open_command, device=self.device)
        expected = (self.num_envs,)
        if ready.shape != expected or open_cmd.shape != expected:
            raise ValueError(f"release_ready/open_command must have shape {expected}")
        if ready.dtype is not torch.bool or open_cmd.dtype is not torch.bool:
            raise TypeError("release_ready/open_command must be torch.bool")

        trigger = ready & open_cmd & ~self._triggered
        self._remaining = torch.where(
            trigger,
            torch.full_like(self._remaining, int(self.cfg.hold_steps)),
            self._remaining,
        )
        self._triggered |= trigger
        return trigger.clone()

    def apply(self, raw_action: Tensor) -> M10PostOpenHoldResult:
        """Apply one countdown step, changing XYZ only in active environments."""
        raw = torch.as_tensor(raw_action)
        if raw.ndim != 2 or raw.shape != (self.num_envs, 7):
            raise ValueError(f"raw_action must have shape ({self.num_envs},7), got {tuple(raw.shape)}")
        if raw.device != self.device:
            raise ValueError(f"raw_action device {raw.device} != tracker device {self.device}")
        if not torch.isfinite(raw).all():
            raise ValueError("raw_action contains NaN/Inf")

        before_remaining = self._remaining.clone()
        active = before_remaining > 0
        out = raw.clone()
        xyz_before = raw[:, :3].clone()
        out[active, :3] = 0.0
        self._remaining = torch.where(active, before_remaining - 1, before_remaining)
        return M10PostOpenHoldResult(
            raw_action=out,
            hold_mask=active.clone(),
            xyz_before=xyz_before,
            xyz_after=out[:, :3].clone(),
            remaining_before=before_remaining,
            remaining_after=self._remaining.clone(),
        )
