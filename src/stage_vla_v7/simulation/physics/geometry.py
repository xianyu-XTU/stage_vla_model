"""Pure tensor geometry used by the physical runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SegmentProjection:
    """Projection of a point onto the infinite line through a segment."""

    alpha: Tensor
    closest: Tensor
    perpendicular_distance: Tensor


def _check_xyz(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {tuple(value.shape)}")


def euclidean_distance(a: Tensor, b: Tensor) -> Tensor:
    _check_xyz("a", a)
    _check_xyz("b", b)
    return torch.linalg.vector_norm(a - b, dim=-1)


def xy_distance(a: Tensor, b: Tensor) -> Tensor:
    _check_xyz("a", a)
    _check_xyz("b", b)
    return torch.linalg.vector_norm(a[..., :2] - b[..., :2], dim=-1)


def midpoint(a: Tensor, b: Tensor) -> Tensor:
    _check_xyz("a", a)
    _check_xyz("b", b)
    return 0.5 * (a + b)


def fingertip_gap(left_tip: Tensor, right_tip: Tensor) -> Tensor:
    return euclidean_distance(left_tip, right_tip)


def absolute_height_error(point: Tensor, reference: Tensor) -> Tensor:
    _check_xyz("point", point)
    _check_xyz("reference", reference)
    return torch.abs(point[..., 2] - reference[..., 2])


def height_aligned(point: Tensor, reference: Tensor, tolerance: float) -> Tensor:
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    return absolute_height_error(point, reference) <= tolerance


def segment_projection(
    point: Tensor,
    start: Tensor,
    end: Tensor,
    *,
    eps: float = 1e-12,
) -> SegmentProjection:
    _check_xyz("point", point)
    _check_xyz("start", start)
    _check_xyz("end", end)
    if eps <= 0:
        raise ValueError("eps must be > 0")

    axis = end - start
    length_sq = torch.sum(axis * axis, dim=-1)
    degenerate = length_sq <= eps
    safe_len = torch.where(degenerate, torch.ones_like(length_sq), length_sq)
    alpha = torch.sum((point - start) * axis, dim=-1) / safe_len
    closest = torch.where(
        degenerate.unsqueeze(-1),
        start,
        start + alpha.unsqueeze(-1) * axis,
    )
    perpendicular_distance = torch.where(
        degenerate,
        euclidean_distance(point, start),
        euclidean_distance(point, closest),
    )
    return SegmentProjection(alpha, closest, perpendicular_distance)


def point_between_fingertips(
    point: Tensor,
    left_tip: Tensor,
    right_tip: Tensor,
    *,
    radial_tolerance: float,
    endpoint_margin: float = 0.0,
) -> Tensor:
    if radial_tolerance < 0:
        raise ValueError("radial_tolerance must be >= 0")
    if not 0.0 <= endpoint_margin < 0.5:
        raise ValueError("endpoint_margin must satisfy 0 <= margin < 0.5")
    projection = segment_projection(point, left_tip, right_tip)
    inside = (
        (projection.alpha >= endpoint_margin)
        & (projection.alpha <= 1.0 - endpoint_margin)
    )
    return inside & (projection.perpendicular_distance <= radial_tolerance)
