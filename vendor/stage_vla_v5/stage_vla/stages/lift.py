"""M6 pure-PyTorch lift detector.

M6 answers:
    "After a stable grasp, has the red cube actually moved upward enough while
     the stable grasp is still valid?"

The detector is intentionally stricter than a bare absolute-z threshold:

    lifted =
        stable_grasp
        AND (red_z_w - reference_red_z_w) >= minimum_object_lift_delta_m

The reference height is captured by the diagnostic/task-state layer at the
first stable-grasp frame. This prevents a pre-grasp bump or a different table
height from silently satisfying the lift milestone.

No permanent history latch lives here. A later success/task-state layer may
store "has_ever_lifted" explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LiftConfig:
    minimum_object_lift_delta_m: float

    def validate(self) -> None:
        if self.minimum_object_lift_delta_m <= 0:
            raise ValueError("minimum_object_lift_delta_m must be > 0")


@dataclass(frozen=True)
class LiftDiagnostics:
    lifted: Tensor
    stable_grasp: Tensor
    lift_delta_m: Tensor
    height_gate: Tensor


def lift_diagnostics(
    red_z_w: Tensor,
    reference_red_z_w: Tensor,
    stable_grasp: Tensor,
    *,
    cfg: LiftConfig,
) -> LiftDiagnostics:
    """Evaluate the vectorized current-frame M6 lift predicate."""
    cfg.validate()

    red_z = torch.as_tensor(red_z_w)
    ref_z = torch.as_tensor(reference_red_z_w, device=red_z.device)
    stable = torch.as_tensor(stable_grasp, device=red_z.device)

    if red_z.shape != ref_z.shape:
        raise ValueError(
            f"red_z_w and reference_red_z_w must share shape; got "
            f"{tuple(red_z.shape)} vs {tuple(ref_z.shape)}"
        )
    if stable.shape != red_z.shape:
        raise ValueError(
            f"stable_grasp shape must match z tensors; got "
            f"{tuple(stable.shape)} vs {tuple(red_z.shape)}"
        )
    if stable.dtype is not torch.bool:
        raise TypeError(f"stable_grasp must be torch.bool, got {stable.dtype}")
    if not torch.isfinite(red_z).all() or not torch.isfinite(ref_z).all():
        raise ValueError("lift z inputs contain NaN/Inf")

    delta = red_z - ref_z
    height_gate = delta >= cfg.minimum_object_lift_delta_m
    lifted = stable & height_gate

    return LiftDiagnostics(
        lifted=lifted,
        stable_grasp=stable.clone(),
        lift_delta_m=delta,
        height_gate=height_gate,
    )


def object_lifted(
    red_z_w: Tensor,
    reference_red_z_w: Tensor,
    stable_grasp: Tensor,
    *,
    cfg: LiftConfig,
) -> Tensor:
    """Return only the vectorized M6 boolean mask."""
    return lift_diagnostics(
        red_z_w,
        reference_red_z_w,
        stable_grasp,
        cfg=cfg,
    ).lifted
