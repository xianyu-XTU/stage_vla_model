"""Pure-PyTorch M10-4 PLACE-stage release credit.

M10-3 learned reach->grasp->lift->transport->PLACE geometry but strict
checkpoint decomposition showed that the policy almost never opened the gripper
post-lift.  M10-4 therefore adds a bounded, one-shot release credit *without*
changing the existing stage machine or M8 potential.

The tracker rewards three bounded events around a verified release-ready PLACE state:
1. the first categorical OPEN command issued from a release-ready *pre-action* state,
2. new progress in actual gripper openness (bounded over the whole episode),
3. the first actual release event (open joints and no physical grasp).

The OPEN-command credit is a one-shot causal bridge added in M10-5 after M10-4
fine-tuning showed stochastic release attempts but no deterministic post-lift
opening. M10-8-T1 corrects its temporal gate: the command is credited according
to release readiness in s_t, while openness/release physics remain evaluated in
s_(t+1). Peak-openness tracking still prevents close/open cycling from farming
state-based credit, and both command/release events are credited at most once.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class M10ReleaseCreditConfig:
    openness_progress_bonus: float
    release_event_bonus: float
    open_token_bonus: float

    def validate(self) -> None:
        vals = torch.tensor(
            [self.openness_progress_bonus, self.release_event_bonus, self.open_token_bonus],
            dtype=torch.float32,
        )
        if not torch.isfinite(vals).all():
            raise ValueError("release credit bonuses must be finite")
        if self.openness_progress_bonus < 0:
            raise ValueError("openness_progress_bonus must be >= 0")
        if self.release_event_bonus < 0:
            raise ValueError("release_event_bonus must be >= 0")
        if self.open_token_bonus < 0:
            raise ValueError("open_token_bonus must be >= 0")


@dataclass(frozen=True)
class M10ReleaseCreditDiagnostics:
    release_ready: Tensor
    open_command_ready: Tensor
    openness: Tensor
    open_command: Tensor
    previous_peak_openness: Tensor
    peak_openness: Tensor
    openness_progress: Tensor
    just_open_command_credited: Tensor
    just_released: Tensor
    command_reward: Tensor
    progress_reward: Tensor
    event_reward: Tensor
    total_reward: Tensor


class M10ReleaseCreditTracker:
    """Bounded per-environment PLACE release credit used by M10-4+.

    Total dense openness credit over one episode is capped by
    ``openness_progress_bonus`` because only increases in the *episode peak*
    openness are rewarded. The OPEN-command and actual-release bonuses can each
    fire at most once, so the full release-specific return remains bounded.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: M10ReleaseCreditConfig,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._peak_openness = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._open_command_credited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._release_credited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def peak_openness(self) -> Tensor:
        return self._peak_openness.clone()

    @property
    def open_command_credited(self) -> Tensor:
        return self._open_command_credited.clone()

    @property
    def release_credited(self) -> Tensor:
        return self._release_credited.clone()

    def _ids(self, env_ids: Sequence[int] | Tensor | slice | None) -> Tensor:
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

    def reset(self, env_ids: Sequence[int] | Tensor | slice | None = None) -> None:
        ids = self._ids(env_ids)
        self._peak_openness[ids] = 0.0
        self._open_command_credited[ids] = False
        self._release_credited[ids] = False

    def update(
        self,
        openness: Tensor,
        release_ready: Tensor,
        released_now: Tensor,
        open_command: Tensor | None = None,
        open_command_ready: Tensor | None = None,
    ) -> M10ReleaseCreditDiagnostics:
        openness_t = torch.as_tensor(openness, dtype=torch.float32, device=self.device)
        ready = torch.as_tensor(release_ready, device=self.device)
        released = torch.as_tensor(released_now, device=self.device)
        if open_command is None:
            open_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            open_cmd = torch.as_tensor(open_command, device=self.device)
        # M10-8-T1 separates action-credit timing from post-step state credit.
        # ``release_ready`` describes s_(t+1) and remains the gate for actual
        # openness/release physics. ``open_command_ready`` describes the
        # pre-action s_t state in which a_t was chosen.  Defaulting to
        # ``release_ready`` preserves compatibility for old pure-unit callers.
        if open_command_ready is None:
            command_ready = ready
        else:
            command_ready = torch.as_tensor(open_command_ready, device=self.device)

        if openness_t.shape != (self.num_envs,):
            raise ValueError(
                f"openness must have shape ({self.num_envs},), got {tuple(openness_t.shape)}"
            )
        if (
            ready.shape != (self.num_envs,)
            or released.shape != (self.num_envs,)
            or open_cmd.shape != (self.num_envs,)
            or command_ready.shape != (self.num_envs,)
        ):
            raise ValueError(
                "release_ready, released_now, open_command and open_command_ready "
                "must have shape [num_envs]"
            )
        if (
            ready.dtype is not torch.bool
            or released.dtype is not torch.bool
            or open_cmd.dtype is not torch.bool
            or command_ready.dtype is not torch.bool
        ):
            raise TypeError(
                "release_ready, released_now, open_command and open_command_ready must be torch.bool"
            )
        if not torch.isfinite(openness_t).all():
            raise ValueError("openness contains NaN/Inf")
        if torch.any((openness_t < 0.0) | (openness_t > 1.0)):
            raise ValueError("openness must lie in [0, 1]")
        if bool((released & ~ready).any().item()):
            raise ValueError("released_now implies release_ready")

        just_open_command = (
            command_ready & open_cmd & ~self._open_command_credited & ~self._release_credited
        )
        command_reward = just_open_command.to(torch.float32) * float(self.cfg.open_token_bonus)

        previous_peak = self._peak_openness.clone()
        active = ready & ~self._release_credited
        candidate_peak = torch.maximum(previous_peak, openness_t)
        new_peak = torch.where(active, candidate_peak, previous_peak)
        progress = torch.where(active, new_peak - previous_peak, torch.zeros_like(openness_t))

        just_released = released & ~self._release_credited
        progress_reward = progress * float(self.cfg.openness_progress_bonus)
        event_reward = just_released.to(torch.float32) * float(self.cfg.release_event_bonus)
        total = command_reward + progress_reward + event_reward

        self._peak_openness.copy_(new_peak)
        self._open_command_credited |= just_open_command
        self._release_credited |= just_released

        return M10ReleaseCreditDiagnostics(
            release_ready=ready.clone(),
            open_command_ready=command_ready.clone(),
            openness=openness_t.clone(),
            open_command=open_cmd.clone(),
            previous_peak_openness=previous_peak,
            peak_openness=new_peak.clone(),
            openness_progress=progress,
            just_open_command_credited=just_open_command,
            just_released=just_released,
            command_reward=command_reward,
            progress_reward=progress_reward,
            event_reward=event_reward,
            total_reward=total,
        )
