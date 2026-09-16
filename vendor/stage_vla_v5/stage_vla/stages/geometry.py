"""Pure PyTorch geometry helpers for stage_vla.

M2 scope:
- No Isaac Sim / Isaac Lab imports.
- No reward functions.
- No stage machine.
- No contact/grasp truth inference.

These helpers only compute geometric quantities that later modules may use as
*dense guidance*. Physical grasp must eventually be gated by real contact data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SegmentProjection:
    """Projection diagnostics for a point relative to a line segment.

    Attributes:
        alpha: Scalar/batched interpolation factor along ``start -> end``.
            0 means ``start`` and 1 means ``end``.
        closest: Closest point on the *infinite line* at ``alpha``.
        perpendicular_distance: Euclidean distance from ``point`` to ``closest``.
    """

    alpha: Tensor
    closest: Tensor
    perpendicular_distance: Tensor


def _check_xyz(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {tuple(value.shape)}")


def euclidean_distance(a: Tensor, b: Tensor) -> Tensor:
    """Return Euclidean distance between batched 3-D points.

    Intentionally uses ``dim=-1`` to avoid the historical ``tensor.norm(-1)``
    bug, where ``-1`` is interpreted as the p-norm order rather than a dimension.
    """

    _check_xyz("a", a)
    _check_xyz("b", b)
    return torch.linalg.vector_norm(a - b, dim=-1)


def xy_distance(a: Tensor, b: Tensor) -> Tensor:
    """Return planar XY distance between batched 3-D points."""

    _check_xyz("a", a)
    _check_xyz("b", b)
    return torch.linalg.vector_norm(a[..., :2] - b[..., :2], dim=-1)


def midpoint(a: Tensor, b: Tensor) -> Tensor:
    """Return midpoint between two batched 3-D points."""

    _check_xyz("a", a)
    _check_xyz("b", b)
    return 0.5 * (a + b)


def fingertip_gap(left_tip: Tensor, right_tip: Tensor) -> Tensor:
    """Return Euclidean distance between left and right fingertip positions."""

    return euclidean_distance(left_tip, right_tip)


def absolute_height_error(point: Tensor, reference: Tensor) -> Tensor:
    """Return ``|point.z - reference.z|``."""

    _check_xyz("point", point)
    _check_xyz("reference", reference)
    return torch.abs(point[..., 2] - reference[..., 2])


def height_aligned(point: Tensor, reference: Tensor, tolerance: float) -> Tensor:
    """Whether two points are vertically aligned within ``tolerance`` meters."""

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
    """Project ``point`` onto the infinite line through ``start -> end``.

    The returned ``alpha`` is not clamped. Therefore:
    - ``0 <= alpha <= 1`` means the projection lies between the endpoints;
    - ``alpha < 0`` lies beyond ``start``;
    - ``alpha > 1`` lies beyond ``end``.

    A degenerate segment is rejected because a fingertip axis with near-zero
    length is not physically meaningful and silently normalizing it would hide
    upstream state errors.
    """

    _check_xyz("point", point)
    _check_xyz("start", start)
    _check_xyz("end", end)
    if eps <= 0:
        raise ValueError("eps must be > 0")

    axis = end - start
    length_sq = torch.sum(axis * axis, dim=-1)
    degenerate = length_sq <= eps
    # A degenerate segment (e.g. gripper fingers fully closed during RL
    # exploration) is handled gracefully: the projection of any point onto a
    # point is that point, with distance |point - start|. Raising on it crashed
    # reward computation mid-training, so it must be a safe value, not an error.
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

    return SegmentProjection(
        alpha=alpha,
        closest=closest,
        perpendicular_distance=perpendicular_distance,
    )


def point_between_fingertips(
    point: Tensor,
    left_tip: Tensor,
    right_tip: Tensor,
    *,
    radial_tolerance: float,
    endpoint_margin: float = 0.0,
) -> Tensor:
    """Pure-geometric test that a point lies close to the fingertip segment.

    This is **not** a physical-grasp detector. It only checks that the point's
    projection lies between the fingertip endpoints and that its perpendicular
    distance to the fingertip axis is small.

    Args:
        point: Object center or other target, shape ``(..., 3)``.
        left_tip: Left fingertip world position, shape ``(..., 3)``.
        right_tip: Right fingertip world position, shape ``(..., 3)``.
        radial_tolerance: Maximum perpendicular distance to the fingertip axis.
        endpoint_margin: Fractional margin in interpolation coordinates. For
            example, 0.1 requires ``alpha`` to lie in ``[0.1, 0.9]``.
    """

    if radial_tolerance < 0:
        raise ValueError("radial_tolerance must be >= 0")
    if not 0.0 <= endpoint_margin < 0.5:
        raise ValueError("endpoint_margin must satisfy 0 <= margin < 0.5")

    proj = segment_projection(point, left_tip, right_tip)
    inside = (proj.alpha >= endpoint_margin) & (proj.alpha <= 1.0 - endpoint_margin)
    near_axis = proj.perpendicular_distance <= radial_tolerance
    return inside & near_axis
