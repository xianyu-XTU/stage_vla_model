"""M10-11-H4X terminal translation hold.

D3 showed that terminal release-ready states still execute non-zero XYZ commands
and produce measurable EE motion.  H4X is a deliberately small intervention:
when the *pre-action* base release-ready truth is active, zero only the raw
pose-IK translation components ``dx/dy/dz``.  GRIP and deterministic yaw remain
untouched.

This module is pure torch so the semantics can be regression-tested without
Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

M10_H4X_TERMINAL_HOLD_SEMANTICS = "pre_action_release_ready_zero_xyz"


@dataclass(frozen=True)
class TerminalXYZHoldResult:
    raw_action: Tensor
    hold_mask: Tensor
    xyz_before: Tensor
    xyz_after: Tensor


def apply_terminal_xyz_hold(raw_action: Tensor, release_ready: Tensor) -> TerminalXYZHoldResult:
    """Zero raw XYZ only where the current pre-action state is release-ready.

    ``raw_action`` is expected to end in the verified 7-D pose-relative IK
    action layout ``[dx,dy,dz,dRx,dRy,dRz,grip]``.  The function intentionally
    does not alter rotation or gripper dimensions.
    """

    raw = torch.as_tensor(raw_action)
    ready = torch.as_tensor(release_ready, device=raw.device)
    if raw.ndim != 2 or raw.shape[-1] != 7:
        raise ValueError(f"raw_action must have shape [N,7], got {tuple(raw.shape)}")
    if ready.shape != (raw.shape[0],):
        raise ValueError(
            f"release_ready must have shape ({raw.shape[0]},), got {tuple(ready.shape)}"
        )
    if ready.dtype != torch.bool:
        raise TypeError(f"release_ready must be bool, got {ready.dtype}")
    if not torch.isfinite(raw).all():
        raise ValueError("raw_action contains NaN/Inf")

    out = raw.clone()
    xyz_before = raw[:, :3].clone()
    out[ready, :3] = 0.0
    xyz_after = out[:, :3].clone()
    return TerminalXYZHoldResult(
        raw_action=out,
        hold_mask=ready.clone(),
        xyz_before=xyz_before,
        xyz_after=xyz_after,
    )
