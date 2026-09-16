"""M10-12-F1 terminal factor-local PPO credit semantics.

H4X showed that physically zeroing terminal XYZ can preserve placement geometry
but destroys the learned OPEN preference.  F1 therefore leaves environment
action execution exactly as T1 and changes only PPO credit assignment:

* non-release-ready transition: standard joint Action-DSL log-probability;
* pre-action base release-ready transition: GRIP-factor log-probability only.

The critic target, scalar reward, entropy and global KL trust-region remain
unchanged.  This module contains only pure PyTorch math so the causal semantics
can be regression-tested without Isaac Lab or RSL-RL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .action_dsl import M9B_CATEGORY_COUNTS, M9B_POLICY_FACTORS
from .factorized_categorical_core import factor_log_prob

M10_F1_CREDIT_SEMANTICS = "pre_action_release_ready_grip_only_ppo_ratio"
M10_F1_READY_EXTRAS_KEY = "m10_factor_local_release_ready"
M10_F1_GRIP_FACTOR_INDEX = 3


@dataclass(frozen=True)
class FactorLocalLogProbSelection:
    current_log_prob: Tensor
    old_log_prob: Tensor
    release_ready: Tensor
    current_factor_log_prob: Tensor
    old_factor_log_prob: Tensor


def _validate_ready_mask(release_ready: Tensor, batch_shape: torch.Size) -> Tensor:
    ready = torch.as_tensor(release_ready)
    if ready.shape == (*batch_shape, 1):
        ready = ready.squeeze(-1)
    if ready.shape != batch_shape:
        raise ValueError(
            f"release_ready must have shape {tuple(batch_shape)} or {tuple(batch_shape) + (1,)}, "
            f"got {tuple(ready.shape)}"
        )
    if ready.dtype != torch.bool:
        if ready.dtype.is_floating_point and torch.all((ready == 0) | (ready == 1)):
            ready = ready.to(torch.bool)
        else:
            raise TypeError(f"release_ready must be bool or exact 0/1 float, got {ready.dtype}")
    return ready


def select_factor_local_log_probs(
    current_logits: Tensor,
    old_logits: Tensor,
    actions: Tensor,
    release_ready: Tensor,
    *,
    category_counts: tuple[int, ...] | list[int] = M9B_CATEGORY_COUNTS,
    grip_factor_index: int = M10_F1_GRIP_FACTOR_INDEX,
) -> FactorLocalLogProbSelection:
    """Select joint-vs-GRIP-only PPO log-probabilities sample by sample.

    For a non-ready sample, the selected log-probability is the sum over all
    Action-DSL factors (the exact M9-B/T1 joint policy likelihood).  For a
    pre-action release-ready sample, only the GRIP factor contributes to the
    selected likelihood and therefore to the direct clipped-policy gradient.
    """

    current = torch.as_tensor(current_logits)
    old = torch.as_tensor(old_logits, device=current.device, dtype=current.dtype)
    action_tensor = torch.as_tensor(actions, device=current.device)
    counts = tuple(int(v) for v in category_counts)
    if len(counts) != M9B_POLICY_FACTORS:
        raise ValueError(f"expected {M9B_POLICY_FACTORS} factors, got {len(counts)}")
    if not (0 <= int(grip_factor_index) < len(counts)):
        raise ValueError(f"invalid grip_factor_index={grip_factor_index}")
    if current.shape != old.shape:
        raise ValueError(f"current/old logits shape mismatch: {tuple(current.shape)} vs {tuple(old.shape)}")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("current/old logits contain NaN/Inf")

    current_factor = factor_log_prob(current, action_tensor, counts)
    old_factor = factor_log_prob(old, action_tensor, counts)
    ready = _validate_ready_mask(release_ready.to(device=current.device), current_factor.shape[:-1])

    current_joint = current_factor.sum(dim=-1)
    old_joint = old_factor.sum(dim=-1)
    current_grip = current_factor[..., grip_factor_index]
    old_grip = old_factor[..., grip_factor_index]

    selected_current = torch.where(ready, current_grip, current_joint)
    selected_old = torch.where(ready, old_grip, old_joint)
    return FactorLocalLogProbSelection(
        current_log_prob=selected_current,
        old_log_prob=selected_old,
        release_ready=ready,
        current_factor_log_prob=current_factor,
        old_factor_log_prob=old_factor,
    )


def clipped_factor_local_surrogate_loss(
    current_logits: Tensor,
    old_logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    release_ready: Tensor,
    *,
    clip_param: float,
    category_counts: tuple[int, ...] | list[int] = M9B_CATEGORY_COUNTS,
    grip_factor_index: int = M10_F1_GRIP_FACTOR_INDEX,
) -> tuple[Tensor, FactorLocalLogProbSelection]:
    """Compute the M10-12-F1 clipped surrogate loss.

    This intentionally changes only the policy likelihood used by the clipped
    surrogate.  Advantage values are the normal scalar PPO advantages produced
    from the unchanged T1 reward and critic.
    """

    if not (0.0 < float(clip_param) < 1.0):
        raise ValueError("clip_param must be in (0,1)")
    selection = select_factor_local_log_probs(
        current_logits,
        old_logits,
        actions,
        release_ready,
        category_counts=category_counts,
        grip_factor_index=grip_factor_index,
    )
    adv = torch.as_tensor(advantages, device=selection.current_log_prob.device)
    if adv.shape == (*selection.current_log_prob.shape, 1):
        adv = adv.squeeze(-1)
    if adv.shape != selection.current_log_prob.shape:
        raise ValueError(
            f"advantages shape must match selected log-prob shape {tuple(selection.current_log_prob.shape)}, "
            f"got {tuple(adv.shape)}"
        )
    if not torch.isfinite(adv).all():
        raise ValueError("advantages contain NaN/Inf")

    ratio = torch.exp(selection.current_log_prob - selection.old_log_prob)
    surrogate = -adv * ratio
    surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - float(clip_param), 1.0 + float(clip_param))
    loss = torch.max(surrogate, surrogate_clipped).mean()
    return loss, selection
