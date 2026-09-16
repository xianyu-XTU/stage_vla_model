"""Pure Torch helpers for scripted-motion convergence diagnostics.

These helpers are diagnostic only. They do not change the controller or task
semantics.
"""

from __future__ import annotations

import torch
from torch import Tensor


def joint_limit_margins(joint_pos: Tensor, joint_pos_limits: Tensor) -> Tensor:
    """Return distance to the nearest lower/upper joint limit.

    Args:
        joint_pos: Joint positions with shape ``(..., J)``.
        joint_pos_limits: Limits with shape ``(..., J, 2)`` where the final
            dimension is ``[lower, upper]``.
    """
    pos = torch.as_tensor(joint_pos)
    limits = torch.as_tensor(joint_pos_limits, device=pos.device, dtype=pos.dtype)
    if limits.shape[:-1] != pos.shape or limits.shape[-1] != 2:
        raise ValueError(
            f"limits must have shape joint_pos.shape + (2,); got "
            f"{tuple(limits.shape)} for {tuple(pos.shape)}"
        )
    if not torch.isfinite(pos).all() or not torch.isfinite(limits).all():
        raise ValueError("joint positions/limits contain NaN/Inf")
    lower = limits[..., 0]
    upper = limits[..., 1]
    if torch.any(upper <= lower):
        raise ValueError("joint upper limits must be greater than lower limits")
    return torch.minimum(pos - lower, upper - pos)


def jacobian_singular_values(jacobian: Tensor) -> Tensor:
    """Return singular values of a batched task-space Jacobian."""
    jac = torch.as_tensor(jacobian)
    if jac.ndim < 2:
        raise ValueError("jacobian must have at least 2 dimensions")
    if not torch.isfinite(jac).all():
        raise ValueError("jacobian contains NaN/Inf")
    return torch.linalg.svdvals(jac)


def jacobian_condition_number(jacobian: Tensor, *, eps: float = 1.0e-9) -> Tensor:
    """Return ``sigma_max / max(sigma_min, eps)`` for each Jacobian."""
    if eps <= 0:
        raise ValueError("eps must be > 0")
    sigma = jacobian_singular_values(jacobian)
    return sigma[..., 0] / torch.clamp(sigma[..., -1], min=eps)
