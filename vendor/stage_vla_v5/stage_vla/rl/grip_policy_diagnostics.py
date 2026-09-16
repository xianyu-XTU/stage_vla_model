"""Pure PyTorch diagnostics for the factorized GRIP policy head.

This module is evaluation-only.  It does not modify actions, rewards, PPO
updates, or the M10 stage machine.  The goal is to measure whether OPEN is
actually becoming the preferred deterministic GRIP action in states that the
M10 reward term already marks as release-ready.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .action_dsl import (
    M9B_CATEGORY_COUNTS,
    M9B_CLOSE_TOKEN,
    M9B_KEEP_TOKEN,
    M9B_OPEN_TOKEN,
)
from .factorized_categorical_core import split_factor_logits


def gripper_logits(
    logits: Tensor,
    category_counts: tuple[int, ...] | list[int] = M9B_CATEGORY_COUNTS,
) -> Tensor:
    """Return the GRIP factor logits from concatenated factorized logits."""
    parts = split_factor_logits(logits, category_counts)
    grip = parts[-1]
    if grip.shape[-1] != 3:
        raise ValueError(f"expected 3 GRIP categories, got {grip.shape[-1]}")
    return grip


def gripper_probabilities(
    logits: Tensor,
    category_counts: tuple[int, ...] | list[int] = M9B_CATEGORY_COUNTS,
) -> Tensor:
    """Return categorical probabilities ordered as OPEN / KEEP / CLOSE."""
    return torch.softmax(gripper_logits(logits, category_counts), dim=-1)


def open_preference_margin(
    logits: Tensor,
    category_counts: tuple[int, ...] | list[int] = M9B_CATEGORY_COUNTS,
) -> Tensor:
    """OPEN logit minus the best non-OPEN logit.

    Positive values mean OPEN is the deterministic argmax.  Negative values
    quantify how far OPEN still is from overtaking KEEP/CLOSE.
    """
    grip = gripper_logits(logits, category_counts)
    open_logit = grip[..., M9B_OPEN_TOKEN]
    non_open = torch.stack(
        (grip[..., M9B_KEEP_TOKEN], grip[..., M9B_CLOSE_TOKEN]), dim=-1
    ).amax(dim=-1)
    return open_logit - non_open


@dataclass(frozen=True)
class GripPreferenceSummary:
    num_envs: int
    ready_samples: int
    ready_envs: int
    argmax_open_samples: int
    argmax_keep_samples: int
    argmax_close_samples: int
    argmax_open_envs: int
    mean_open_prob: float
    mean_keep_prob: float
    mean_close_prob: float
    max_open_prob: float
    mean_open_margin: float
    max_open_margin: float

    @property
    def open_argmax_fraction(self) -> float:
        if self.ready_samples == 0:
            return 0.0
        return self.argmax_open_samples / self.ready_samples


class GripPreferenceTracker:
    """Streaming release-ready GRIP preference statistics."""

    def __init__(self, num_envs: int, device: str | torch.device) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.ready_env_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.open_argmax_env_seen = torch.zeros_like(self.ready_env_seen)
        self.ready_samples = 0
        self.argmax_counts = torch.zeros(3, dtype=torch.long, device=self.device)
        self.prob_sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.max_open_prob = float("-inf")
        self.margin_sum = 0.0
        self.max_open_margin = float("-inf")

    def update(
        self,
        logits: Tensor,
        release_ready: Tensor,
        *,
        deterministic_tokens: Tensor | None = None,
    ) -> None:
        value = torch.as_tensor(logits, device=self.device)
        ready = torch.as_tensor(release_ready, device=self.device, dtype=torch.bool).reshape(-1)
        if value.ndim != 2 or value.shape[0] != self.num_envs:
            raise ValueError(
                f"logits must have shape ({self.num_envs}, total_logits), got {tuple(value.shape)}"
            )
        if ready.shape != (self.num_envs,):
            raise ValueError(
                f"release_ready must have shape ({self.num_envs},), got {tuple(ready.shape)}"
            )
        if not bool(ready.any().item()):
            return

        grip = gripper_logits(value)
        probs = torch.softmax(grip, dim=-1)
        argmax = torch.argmax(grip, dim=-1)

        if deterministic_tokens is not None:
            tokens = torch.as_tensor(deterministic_tokens, device=self.device)
            if tokens.shape != (self.num_envs, 4):
                raise ValueError(
                    f"deterministic_tokens must have shape ({self.num_envs}, 4), got {tuple(tokens.shape)}"
                )
            selected = torch.round(tokens[..., 3]).to(torch.long)
            if not torch.equal(selected[ready], argmax[ready]):
                raise RuntimeError(
                    "deterministic GRIP tokens disagree with raw-logit argmax on release-ready states"
                )

        ready_idx = ready.nonzero(as_tuple=False).flatten()
        ready_grip = argmax[ready]
        ready_probs = probs[ready]
        margins = open_preference_margin(value)[ready]

        self.ready_env_seen[ready_idx] = True
        open_ready = ready_grip == M9B_OPEN_TOKEN
        if bool(open_ready.any().item()):
            self.open_argmax_env_seen[ready_idx[open_ready]] = True

        self.ready_samples += int(ready_idx.numel())
        for token in (M9B_OPEN_TOKEN, M9B_KEEP_TOKEN, M9B_CLOSE_TOKEN):
            self.argmax_counts[token] += (ready_grip == token).sum()
        self.prob_sums += ready_probs.to(torch.float64).sum(dim=0)
        self.max_open_prob = max(
            self.max_open_prob,
            float(ready_probs[:, M9B_OPEN_TOKEN].max().item()),
        )
        self.margin_sum += float(margins.to(torch.float64).sum().item())
        self.max_open_margin = max(self.max_open_margin, float(margins.max().item()))

    def summary(self) -> GripPreferenceSummary:
        if self.ready_samples == 0:
            return GripPreferenceSummary(
                num_envs=self.num_envs,
                ready_samples=0,
                ready_envs=0,
                argmax_open_samples=0,
                argmax_keep_samples=0,
                argmax_close_samples=0,
                argmax_open_envs=0,
                mean_open_prob=0.0,
                mean_keep_prob=0.0,
                mean_close_prob=0.0,
                max_open_prob=0.0,
                mean_open_margin=0.0,
                max_open_margin=0.0,
            )
        probs = self.prob_sums / float(self.ready_samples)
        return GripPreferenceSummary(
            num_envs=self.num_envs,
            ready_samples=self.ready_samples,
            ready_envs=int(self.ready_env_seen.sum().item()),
            argmax_open_samples=int(self.argmax_counts[M9B_OPEN_TOKEN].item()),
            argmax_keep_samples=int(self.argmax_counts[M9B_KEEP_TOKEN].item()),
            argmax_close_samples=int(self.argmax_counts[M9B_CLOSE_TOKEN].item()),
            argmax_open_envs=int(self.open_argmax_env_seen.sum().item()),
            mean_open_prob=float(probs[M9B_OPEN_TOKEN].item()),
            mean_keep_prob=float(probs[M9B_KEEP_TOKEN].item()),
            mean_close_prob=float(probs[M9B_CLOSE_TOKEN].item()),
            max_open_prob=self.max_open_prob,
            mean_open_margin=self.margin_sum / float(self.ready_samples),
            max_open_margin=self.max_open_margin,
        )
