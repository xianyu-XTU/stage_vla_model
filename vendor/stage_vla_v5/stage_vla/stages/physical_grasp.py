"""M4 pure-PyTorch current-frame physical-grasp detector.

M3.2-net already validated the underlying evidence on the user's machine.
M4 formalizes that evidence as a reusable, vectorized, side-effect-free predicate.

physical_grasp =
    red cube between true fingertip frames
    AND left fingertip height aligned
    AND right fingertip height aligned
    AND finger-A net force >= threshold
    AND finger-B net force >= threshold

Temporal persistence is intentionally excluded; that is M5 stable_grasp.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .geometry import absolute_height_error, fingertip_gap, segment_projection


@dataclass(frozen=True)
class PhysicalGraspConfig:
    radial_tolerance_m: float
    height_tolerance_m: float
    contact_force_threshold_n: float
    endpoint_margin: float

    def validate(self) -> None:
        if self.radial_tolerance_m < 0:
            raise ValueError("radial_tolerance_m must be >= 0")
        if self.height_tolerance_m < 0:
            raise ValueError("height_tolerance_m must be >= 0")
        if self.contact_force_threshold_n < 0:
            raise ValueError("contact_force_threshold_n must be >= 0")
        if not 0.0 <= self.endpoint_margin < 0.5:
            raise ValueError("endpoint_margin must satisfy 0 <= margin < 0.5")


@dataclass(frozen=True)
class PhysicalGraspDiagnostics:
    physical_grasp: Tensor
    between_fingertips: Tensor
    left_height_aligned: Tensor
    right_height_aligned: Tensor
    finger_a_contact: Tensor
    finger_b_contact: Tensor
    projection_alpha: Tensor
    radial_error_m: Tensor
    left_height_error_m: Tensor
    right_height_error_m: Tensor
    fingertip_gap_m: Tensor
    finger_a_force_n: Tensor
    finger_b_force_n: Tensor


def _force_magnitude(name: str, value: Tensor) -> Tensor:
    value = torch.as_tensor(value)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if torch.any(value < 0):
        raise ValueError(f"{name} must be a non-negative magnitude")
    return value


def physical_grasp_diagnostics(
    red_pos_w: Tensor,
    left_tip_w: Tensor,
    right_tip_w: Tensor,
    finger_a_force_n: Tensor,
    finger_b_force_n: Tensor,
    *,
    cfg: PhysicalGraspConfig,
    reference_pos_w: Tensor | None = None,
    height_tolerance_m: Tensor | float | None = None,
) -> PhysicalGraspDiagnostics:
    cfg.validate()
    reference = red_pos_w if reference_pos_w is None else reference_pos_w

    projection = segment_projection(reference, left_tip_w, right_tip_w)
    inside = (
        (projection.alpha >= cfg.endpoint_margin)
        & (projection.alpha <= 1.0 - cfg.endpoint_margin)
    )
    between = inside & (
        projection.perpendicular_distance <= cfg.radial_tolerance_m
    )

    left_height_error = absolute_height_error(left_tip_w, reference)
    right_height_error = absolute_height_error(right_tip_w, reference)
    height_tolerance = torch.as_tensor(
        cfg.height_tolerance_m if height_tolerance_m is None else height_tolerance_m,
        device=left_height_error.device,
        dtype=left_height_error.dtype,
    )
    if not torch.isfinite(height_tolerance).all() or torch.any(height_tolerance < 0):
        raise ValueError("height_tolerance_m must be finite and non-negative")
    try:
        height_tolerance = torch.broadcast_to(
            height_tolerance, left_height_error.shape
        )
    except RuntimeError as exc:
        raise ValueError("height_tolerance_m is not broadcastable to the batch") from exc
    left_height_ok = left_height_error <= height_tolerance
    right_height_ok = right_height_error <= height_tolerance

    force_a = _force_magnitude("finger_a_force_n", finger_a_force_n)
    force_b = _force_magnitude("finger_b_force_n", finger_b_force_n)
    contact_a = force_a >= cfg.contact_force_threshold_n
    contact_b = force_b >= cfg.contact_force_threshold_n

    grasp = between & left_height_ok & right_height_ok & contact_a & contact_b

    return PhysicalGraspDiagnostics(
        physical_grasp=grasp,
        between_fingertips=between,
        left_height_aligned=left_height_ok,
        right_height_aligned=right_height_ok,
        finger_a_contact=contact_a,
        finger_b_contact=contact_b,
        projection_alpha=projection.alpha,
        radial_error_m=projection.perpendicular_distance,
        left_height_error_m=left_height_error,
        right_height_error_m=right_height_error,
        fingertip_gap_m=fingertip_gap(left_tip_w, right_tip_w),
        finger_a_force_n=force_a,
        finger_b_force_n=force_b,
    )


def physical_grasp(
    red_pos_w: Tensor,
    left_tip_w: Tensor,
    right_tip_w: Tensor,
    finger_a_force_n: Tensor,
    finger_b_force_n: Tensor,
    *,
    cfg: PhysicalGraspConfig,
) -> Tensor:
    return physical_grasp_diagnostics(
        red_pos_w,
        left_tip_w,
        right_tip_w,
        finger_a_force_n,
        finger_b_force_n,
        cfg=cfg,
    ).physical_grasp
