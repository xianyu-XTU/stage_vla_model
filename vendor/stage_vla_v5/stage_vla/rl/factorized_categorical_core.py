"""Pure PyTorch math for M9-B factorized categorical actions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical


def validate_category_counts(category_counts: tuple[int, ...] | list[int], output_dim: int) -> tuple[int, ...]:
    counts = tuple(int(v) for v in category_counts)
    if len(counts) != int(output_dim):
        raise ValueError("category_counts length must match output_dim")
    if any(v < 2 for v in counts):
        raise ValueError("each categorical factor must contain at least two categories")
    return counts


def split_factor_logits(logits: Tensor, category_counts: tuple[int, ...] | list[int]) -> tuple[Tensor, ...]:
    counts = tuple(int(v) for v in category_counts)
    expected = int(sum(counts))
    if logits.ndim < 1 or logits.shape[-1] != expected:
        raise ValueError(f"expected last logit dimension {expected}, got {tuple(logits.shape)}")
    return tuple(torch.split(logits, counts, dim=-1))


def deterministic_factor_tokens(logits: Tensor, category_counts: tuple[int, ...] | list[int]) -> Tensor:
    return torch.stack([torch.argmax(part, dim=-1) for part in split_factor_logits(logits, category_counts)], dim=-1)


def validate_factor_tokens(actions: Tensor, category_counts: tuple[int, ...] | list[int]) -> Tensor:
    value = torch.as_tensor(actions)
    counts = tuple(int(v) for v in category_counts)
    if value.ndim < 1 or value.shape[-1] != len(counts):
        raise ValueError(f"actions must end in {len(counts)} factors, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError("categorical actions contain NaN/Inf")
    rounded = torch.round(value)
    if not torch.allclose(value, rounded, atol=1e-6, rtol=0.0):
        raise ValueError("categorical actions must be integer-valued token indices")
    idx = rounded.to(torch.long)
    for factor, count in enumerate(counts):
        token = idx[..., factor]
        if bool(((token < 0) | (token >= count)).any().item()):
            raise ValueError(f"factor {factor} token outside [0,{count - 1}]")
    return idx


def factor_log_prob(logits: Tensor, actions: Tensor, category_counts: tuple[int, ...] | list[int]) -> Tensor:
    """Return one log-probability per categorical factor.

    Output shape is ``actions.shape[:-1] + (num_factors,)``.  Keeping factor
    terms separate is required by M10-12-F1 so terminal release-ready samples
    can apply PPO credit only to the GRIP factor without changing the Action-DSL
    vocabulary or environment action execution.
    """
    parts = split_factor_logits(logits, category_counts)
    idx = validate_factor_tokens(actions, category_counts)
    values = [Categorical(logits=part).log_prob(idx[..., i]) for i, part in enumerate(parts)]
    return torch.stack(values, dim=-1)


def joint_log_prob(logits: Tensor, actions: Tensor, category_counts: tuple[int, ...] | list[int]) -> Tensor:
    return factor_log_prob(logits, actions, category_counts).sum(dim=-1)


def joint_entropy(logits: Tensor, category_counts: tuple[int, ...] | list[int]) -> Tensor:
    values = [Categorical(logits=part).entropy() for part in split_factor_logits(logits, category_counts)]
    return torch.stack(values, dim=-1).sum(dim=-1)


def factor_index_mean_std(logits: Tensor, category_counts: tuple[int, ...] | list[int]) -> tuple[Tensor, Tensor]:
    means = []
    stds = []
    for part in split_factor_logits(logits, category_counts):
        probs = torch.softmax(part, dim=-1)
        values = torch.arange(part.shape[-1], device=part.device, dtype=part.dtype)
        mean = (probs * values).sum(dim=-1)
        var = (probs * (values - mean.unsqueeze(-1)).square()).sum(dim=-1)
        means.append(mean)
        stds.append(torch.sqrt(torch.clamp_min(var, 0.0)))
    return torch.stack(means, dim=-1), torch.stack(stds, dim=-1)


def joint_kl(
    old_logits: Tensor,
    new_logits: Tensor,
    category_counts: tuple[int, ...] | list[int],
) -> Tensor:
    old_parts = split_factor_logits(old_logits, category_counts)
    new_parts = split_factor_logits(new_logits, category_counts)
    values = [
        torch.distributions.kl_divergence(Categorical(logits=old), Categorical(logits=new))
        for old, new in zip(old_parts, new_parts, strict=True)
    ]
    return torch.stack(values, dim=-1).sum(dim=-1)
