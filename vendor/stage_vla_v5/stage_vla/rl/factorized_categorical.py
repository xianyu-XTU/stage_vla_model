"""RSL-RL 5.x factorized categorical distribution for the M9-B Action DSL.

RSL-RL 5 exposes a distribution extension API through ``Distribution``.  The
stock library distributions are continuous (Gaussian / heteroscedastic
Gaussian / Beta), so M9-B supplies a project-local distribution with four
independent categorical factors.  PPO still receives the correct joint
log-probability, entropy, and KL divergence by summing the four factor terms.

This file intentionally imports RSL-RL.  Pure Action-DSL semantics live in
``action_dsl.py`` and remain unit-testable without Isaac/RSL-RL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from rsl_rl.modules.distribution import Distribution

from .action_dsl import M9B_CATEGORY_COUNTS, M9B_POLICY_FACTORS
from .factorized_categorical_core import (
    deterministic_factor_tokens,
    factor_index_mean_std,
    joint_entropy,
    joint_kl,
    joint_log_prob,
    split_factor_logits,
    validate_category_counts,
)


class FactorizedCategoricalDistribution(Distribution):
    """Independent categorical factors with a summed joint PPO log-probability."""

    def __init__(
        self,
        output_dim: int,
        category_counts: tuple[int, ...] | list[int] = M9B_CATEGORY_COUNTS,
    ) -> None:
        super().__init__(output_dim)
        if output_dim != M9B_POLICY_FACTORS:
            raise ValueError(f"M9-B expects output_dim={M9B_POLICY_FACTORS}, got {output_dim}")
        self.category_counts = validate_category_counts(category_counts, output_dim)
        self._total_logits = int(sum(self.category_counts))
        self._distributions: tuple[Categorical, ...] | None = None
        self._raw_logits: Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self._total_logits

    def _split_logits(self, mlp_output: Tensor) -> tuple[Tensor, ...]:
        return split_factor_logits(mlp_output, self.category_counts)

    def update(self, mlp_output: Tensor) -> None:
        self._raw_logits = mlp_output
        self._distributions = tuple(Categorical(logits=part) for part in self._split_logits(mlp_output))

    def _require_distributions(self) -> tuple[Categorical, ...]:
        if self._distributions is None:
            raise RuntimeError("distribution.update() must be called before sampling/statistics")
        return self._distributions

    def sample(self) -> Tensor:
        samples = [dist.sample() for dist in self._require_distributions()]
        # RSL-RL rollout storage allocates floating action buffers.  Integer-valued
        # floating tokens preserve compatibility while log_prob() casts back to long.
        dtype = self._raw_logits.dtype if self._raw_logits is not None else torch.float32
        return torch.stack(samples, dim=-1).to(dtype=dtype)

    def deterministic_output(self, mlp_output: Tensor) -> Tensor:
        return deterministic_factor_tokens(mlp_output, self.category_counts).to(dtype=mlp_output.dtype)

    def as_deterministic_output_module(self) -> nn.Module:
        return _FactorizedArgmax(self.category_counts)

    @property
    def mean(self) -> Tensor:
        if self._raw_logits is None:
            raise RuntimeError("distribution.update() must be called before mean")
        return factor_index_mean_std(self._raw_logits, self.category_counts)[0]

    @property
    def std(self) -> Tensor:
        if self._raw_logits is None:
            raise RuntimeError("distribution.update() must be called before std")
        return factor_index_mean_std(self._raw_logits, self.category_counts)[1]

    @property
    def entropy(self) -> Tensor:
        if self._raw_logits is None:
            raise RuntimeError("distribution.update() must be called before entropy")
        return joint_entropy(self._raw_logits, self.category_counts)

    @property
    def params(self) -> tuple[Tensor, ...]:
        if self._raw_logits is None:
            raise RuntimeError("distribution.update() must be called before params")
        # One concatenated tensor keeps RolloutStorage simple and reconstructs all
        # four independent factors exactly for adaptive-KL PPO.
        return (self._raw_logits,)

    def log_prob(self, outputs: Tensor) -> Tensor:
        if self._raw_logits is None:
            raise RuntimeError("distribution.update() must be called before log_prob")
        return joint_log_prob(self._raw_logits, outputs, self.category_counts)

    def kl_divergence(self, old_params: tuple[Tensor, ...], new_params: tuple[Tensor, ...]) -> Tensor:
        if len(old_params) != 1 or len(new_params) != 1:
            raise ValueError("FactorizedCategoricalDistribution expects one concatenated logits tensor")
        return joint_kl(old_params[0], new_params[0], self.category_counts)



class _FactorizedArgmax(nn.Module):
    """Export-friendly deterministic token decoder for the actor logits."""

    def __init__(self, category_counts: tuple[int, ...]) -> None:
        super().__init__()
        self.category_counts = category_counts

    def forward(self, mlp_output: Tensor) -> Tensor:
        parts = torch.split(mlp_output, self.category_counts, dim=-1)
        tokens = [torch.argmax(part, dim=-1) for part in parts]
        return torch.stack(tokens, dim=-1).to(dtype=mlp_output.dtype)
