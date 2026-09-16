"""M9-B factorized Action-DSL semantics and deterministic decoder.

The policy emits four categorical token indices::

    [DX, DY, DZ, GRIP]

Translation factors each have five categories mapping to symbolic bins
``{-2,-1,0,+1,+2}``.  To keep the low-level control envelope identical to the
M9-A continuous baseline, bins are normalized by the largest absolute bin, so
``±2`` maps to raw ``±1`` and therefore to the same configured ``±0.004 m``
processed IK translation range.

The gripper factor has three categories::

    0 -> OPEN
    1 -> KEEP
    2 -> CLOSE

``KEEP`` repeats the previous decoded binary gripper command.  The memory is
reset to OPEN after an episode reset.  This temporal state belongs only to the
Action-DSL decoder; it is not stage state and it does not alter the reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

M9B_POLICY_FACTORS = 4
M9B_TRANSLATION_CATEGORIES = 5
M9B_GRIPPER_CATEGORIES = 3
M9B_CATEGORY_COUNTS = (
    M9B_TRANSLATION_CATEGORIES,
    M9B_TRANSLATION_CATEGORIES,
    M9B_TRANSLATION_CATEGORIES,
    M9B_GRIPPER_CATEGORIES,
)

M9B_OPEN_TOKEN = 0
M9B_KEEP_TOKEN = 1
M9B_CLOSE_TOKEN = 2
M9B_TRANSLATION_CENTER_TOKEN = 2
M9B_TRANSLATION_MAX_ABS_BIN = 2


@dataclass(frozen=True)
class M9BDecodedAction:
    """Result of decoding one batch of factorized Action-DSL tokens."""

    token_indices: Tensor
    translation_bins: Tensor
    logical_action: Tensor
    next_gripper_raw: Tensor


def _as_integral_token_tensor(tokens: Tensor) -> Tensor:
    value = torch.as_tensor(tokens)
    if value.ndim < 1 or value.shape[-1] != M9B_POLICY_FACTORS:
        raise ValueError(
            f"M9-B token action must end in {M9B_POLICY_FACTORS} factors, got shape {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("M9-B token action contains NaN/Inf")
    rounded = torch.round(value)
    if not torch.allclose(value, rounded, atol=1e-6, rtol=0.0):
        raise ValueError("M9-B token actions must be integer-valued category indices")
    return rounded.to(dtype=torch.long)


def validate_m9b_tokens(tokens: Tensor) -> Tensor:
    """Validate and return integral ``[...,4]`` token indices."""
    idx = _as_integral_token_tensor(tokens)
    counts = M9B_CATEGORY_COUNTS
    for factor, count in enumerate(counts):
        t = idx[..., factor]
        if bool(((t < 0) | (t >= count)).any().item()):
            raise ValueError(
                f"M9-B factor {factor} token out of range [0,{count - 1}]: "
                f"min={int(t.min().item())}, max={int(t.max().item())}"
            )
    return idx


def translation_token_to_bin(token_indices: Tensor) -> Tensor:
    """Map translation category indices ``0..4`` to symbolic bins ``-2..+2``."""
    idx = torch.as_tensor(token_indices)
    if idx.ndim < 1 or idx.shape[-1] != 3:
        raise ValueError(f"translation token tensor must end in 3 values, got {tuple(idx.shape)}")
    if bool(((idx < 0) | (idx >= M9B_TRANSLATION_CATEGORIES)).any().item()):
        raise ValueError("translation token index must be in [0,4]")
    return idx.to(torch.long) - M9B_TRANSLATION_CENTER_TOKEN


def translation_bin_to_raw(translation_bins: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """Map symbolic ``-2..+2`` bins to the M9-A-compatible raw range ``-1..+1``."""
    bins = torch.as_tensor(translation_bins)
    if bins.ndim < 1 or bins.shape[-1] != 3:
        raise ValueError(f"translation bins must end in 3 values, got {tuple(bins.shape)}")
    if bool((bins.abs() > M9B_TRANSLATION_MAX_ABS_BIN).any().item()):
        raise ValueError("translation bins must stay inside [-2,+2]")
    out_dtype = dtype if dtype is not None else (bins.dtype if bins.is_floating_point() else torch.float32)
    return bins.to(dtype=out_dtype) / float(M9B_TRANSLATION_MAX_ABS_BIN)


def decode_m9b_tokens(tokens: Tensor, previous_gripper_raw: Tensor) -> M9BDecodedAction:
    """Decode factorized tokens to M9-A-compatible logical ``[dx,dy,dz,grip]``.

    ``previous_gripper_raw`` has shape equal to ``tokens.shape[:-1]`` and must
    contain binary commands ``+1`` (open) or ``-1`` (close).  KEEP preserves
    that command.
    """
    value = torch.as_tensor(tokens)
    idx = validate_m9b_tokens(value)
    previous = torch.as_tensor(previous_gripper_raw, device=value.device)
    expected_shape = value.shape[:-1]
    if previous.shape != expected_shape:
        raise ValueError(
            f"previous_gripper_raw must have shape {tuple(expected_shape)}, got {tuple(previous.shape)}"
        )
    if not torch.isfinite(previous).all():
        raise ValueError("previous_gripper_raw contains NaN/Inf")
    previous_float = previous.to(dtype=value.dtype if value.is_floating_point() else torch.float32)
    valid_prev = torch.isclose(previous_float.abs(), torch.ones_like(previous_float), atol=1e-6, rtol=0.0)
    if not bool(valid_prev.all().item()):
        raise ValueError("previous_gripper_raw must contain only -1 or +1")

    bins = translation_token_to_bin(idx[..., :3])
    raw_xyz = translation_bin_to_raw(bins, dtype=previous_float.dtype)

    grip_token = idx[..., 3]
    next_grip = previous_float.clone()
    next_grip = torch.where(grip_token == M9B_OPEN_TOKEN, torch.ones_like(next_grip), next_grip)
    next_grip = torch.where(grip_token == M9B_CLOSE_TOKEN, -torch.ones_like(next_grip), next_grip)
    # KEEP intentionally leaves next_grip unchanged.

    logical = torch.empty((*expected_shape, M9B_POLICY_FACTORS), device=value.device, dtype=previous_float.dtype)
    logical[..., :3] = raw_xyz
    logical[..., 3] = next_grip

    return M9BDecodedAction(
        token_indices=idx,
        translation_bins=bins,
        logical_action=logical,
        next_gripper_raw=next_grip,
    )
