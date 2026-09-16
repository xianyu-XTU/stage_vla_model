"""Temporal downstream-policy feasibility for TRANSPORT -> frozen PLACE.

Unlike V3's geometric ready classifier, the target here is measured by rolling
out the frozen downstream policy to task completion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def beta_smoothed_success_probability(
    success_count: torch.Tensor,
    rollout_count: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Return a stable soft target from repeated downstream rollouts."""
    success = torch.as_tensor(success_count, dtype=torch.float32)
    total = torch.as_tensor(rollout_count, dtype=torch.float32)
    if torch.any(total <= 0):
        raise ValueError("rollout_count must be positive")
    if torch.any(success < 0) or torch.any(success > total):
        raise ValueError("success_count must lie in [0, rollout_count]")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    return (success + float(alpha)) / (total + float(alpha) + float(beta))


class TransitionFeasibilityNet(nn.Module):
    """Predict the probability that frozen A1 succeeds from a state history."""

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("state_dim, hidden_dim and num_layers must be positive")
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(state_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, state_dim]")
        output, _ = self.gru(history)
        return self.head(output[:, -1]).squeeze(-1)

    @torch.inference_mode()
    def predict_probability(self, history: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(history))


def feasibility_loss(logits: torch.Tensor, target_probability: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(target_probability, dtype=logits.dtype, device=logits.device)
    if logits.shape != target.shape:
        raise ValueError("logits and target_probability must have matching shape")
    if torch.any(target < 0) or torch.any(target > 1):
        raise ValueError("target_probability must lie in [0, 1]")
    return F.binary_cross_entropy_with_logits(logits, target)


def compose_feasibility_reward(
    base_reward: torch.Tensor,
    previous_probability: torch.Tensor,
    next_probability: torch.Tensor,
    *,
    progress_weight: float,
    handoff: torch.Tensor | None = None,
    handoff_bonus_weight: float = 0.0,
) -> torch.Tensor:
    """Reward improvement in downstream feasibility and high-quality handoff."""
    shaped = base_reward + float(progress_weight) * (next_probability - previous_probability)
    if handoff is not None:
        shaped = shaped + float(handoff_bonus_weight) * handoff.float() * next_probability
    return shaped


@dataclass
class FeasibilityGate:
    """Require sustained downstream feasibility before switching to A1."""

    num_envs: int
    threshold: float = 0.70
    consecutive_steps: int = 3
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.consecutive_steps <= 0:
            raise ValueError("num_envs and consecutive_steps must be positive")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        self.device = torch.device(self.device)
        self.count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.count.zero_()
            self.latched.zero_()
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.count[ids] = 0
        self.latched[ids] = False

    def update(self, probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.as_tensor(probability, dtype=torch.float32, device=self.device)
        if probability.shape != (self.num_envs,):
            raise ValueError(f"probability must have shape ({self.num_envs},)")
        ready = probability >= float(self.threshold)
        active = ~self.latched
        self.count = torch.where(
            active & ready,
            self.count + 1,
            torch.where(active, torch.zeros_like(self.count), self.count),
        )
        newly_latched = active & (self.count >= int(self.consecutive_steps))
        self.latched |= newly_latched
        return self.latched.clone(), newly_latched
