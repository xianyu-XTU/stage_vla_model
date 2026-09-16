"""M9-A logical-action adapter.

The learned policy intentionally exposes only four controls::

    [dx, dy, dz, gripper]

The underlying Franka task keeps pose-relative Differential IK, whose raw
command is seven dimensional::

    [dx, dy, dz, dRx, dRy, dRz, gripper]

M9-A2-R2 keeps dRx/dRy deterministic zero and injects a deterministic dRz
computed from the red-cube edge alignment controller.  Thus yaw alignment is
shared low-level geometry, not a fifth learned policy degree of freedom.
"""

from __future__ import annotations

import torch
from torch import Tensor

M9A_POLICY_ACTION_DIM = 4
M9A_RAW_ACTION_DIM = 7


def expand_m9a_policy_action(
    actions: Tensor,
    *,
    yaw_delta_raw: Tensor | None = None,
) -> Tensor:
    """Expand ``[...,4]`` policy actions to ``[...,7]`` pose-IK actions.

    Default mapping (for pure tests/backward compatibility)::

        [dx, dy, dz, grip] -> [dx, dy, dz, 0, 0, 0, grip]

    With deterministic edge alignment::

        [dx, dy, dz, grip] -> [dx, dy, dz, 0, 0, yaw_delta_raw, grip]

    ``yaw_delta_raw`` is a *raw* action value.  The Isaac action term later
    multiplies it by the configured rotation scale before the Differential IK
    controller interprets it as an angle-axis dRz pose delta.
    """
    value = torch.as_tensor(actions)
    if value.ndim < 1 or value.shape[-1] != M9A_POLICY_ACTION_DIM:
        raise ValueError(
            f"M9-A policy action must end in {M9A_POLICY_ACTION_DIM} values, "
            f"got shape {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("M9-A policy action contains NaN/Inf")

    raw = torch.zeros((*value.shape[:-1], M9A_RAW_ACTION_DIM), dtype=value.dtype, device=value.device)
    raw[..., 0:3] = value[..., 0:3]
    raw[..., 6] = value[..., 3]

    if yaw_delta_raw is not None:
        yaw = torch.as_tensor(yaw_delta_raw, dtype=value.dtype, device=value.device)
        expected = value.shape[:-1]
        if yaw.shape != expected:
            raise ValueError(
                f"yaw_delta_raw must have shape {tuple(expected)}, got {tuple(yaw.shape)}"
            )
        if not torch.isfinite(yaw).all():
            raise ValueError("yaw_delta_raw contains NaN/Inf")
        raw[..., 5] = yaw

    return raw
