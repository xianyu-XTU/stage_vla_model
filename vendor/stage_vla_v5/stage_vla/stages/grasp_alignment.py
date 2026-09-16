"""Pure geometry helpers for gripper/cube planar grasp alignment.

For the parallel Franka gripper, the segment joining the two true fingertip
frames approximates the planar closing axis.  For an upright square cube, the
most forgiving top-down grasp aligns this axis with one of the cube's local
X/Y axes so the fingers close on a pair of opposite faces.  A 45-degree axis
instead spans the face diagonal and leaves much less planar clearance.

This module is pure PyTorch and intentionally contains no Isaac Lab imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GraspAlignmentDiagnostics:
    closing_axis_xy: Tensor
    cube_x_axis_xy: Tensor
    cube_y_axis_xy: Tensor
    angle_to_cube_x_deg: Tensor
    angle_to_cube_y_deg: Tensor
    nearest_edge_angle_deg: Tensor
    projected_cube_width_m: Tensor


@dataclass(frozen=True)
class EdgeAlignmentCommand:
    """Signed shortest planar correction toward a cube edge axis."""

    signed_error_rad: Tensor
    bounded_step_rad: Tensor
    target_axis_xy: Tensor
    target_axis_index: Tensor
    nearest_edge_angle_deg: Tensor


def _as_xy(value: Tensor, *, name: str) -> Tensor:
    value = torch.as_tensor(value)
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError(f"{name} must end in 2 XY values, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _unit_xy(value: Tensor, *, name: str, eps: float = 1e-9) -> Tensor:
    value = _as_xy(value, name=name)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    degenerate = norm <= eps
    # A degenerate planar direction (e.g. gripper fully closed so the fingertip
    # closing axis has zero XY length) is handled gracefully by falling back to
    # the +X axis. Raising here crashed reward/action computation mid-RL when
    # exploration closed the gripper; a default direction is harmless because a
    # degenerate closing axis carries no alignment information.
    safe_norm = torch.where(degenerate, torch.ones_like(norm), norm)
    unit = value / safe_norm
    fallback = torch.tensor([1.0, 0.0], device=value.device, dtype=value.dtype)
    return torch.where(degenerate.broadcast_to(unit.shape), fallback, unit)


def undirected_angle_deg(a_xy: Tensor, b_xy: Tensor) -> Tensor:
    """Angle in degrees between two *undirected* planar axes, in [0, 90]."""
    a = _unit_xy(a_xy, name="a_xy")
    b = _unit_xy(b_xy, name="b_xy")
    dot = torch.sum(a * b, dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def _signed_angle_rad(a_xy: Tensor, b_xy: Tensor) -> Tensor:
    """Signed CCW angle from directed axis ``a`` to ``b`` in [-pi, pi]."""
    a = _unit_xy(a_xy, name="a_xy")
    b = _unit_xy(b_xy, name="b_xy")
    cross_z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = torch.sum(a * b, dim=-1)
    return torch.atan2(cross_z, dot)


def square_projected_width_m(
    closing_axis_xy: Tensor,
    cube_x_axis_xy: Tensor,
    cube_y_axis_xy: Tensor,
    *,
    cube_side_m: float,
) -> Tensor:
    """Projected width of a square face along the gripper closing axis.

    For an upright square of side ``s`` and orthonormal in-plane axes x/y,
    width along unit direction u is ``s*(|u.x| + |u.y|)`` in the cube basis.
    It ranges from ``s`` (edge-aligned) to ``s*sqrt(2)`` (diagonal).
    """
    if cube_side_m <= 0:
        raise ValueError("cube_side_m must be > 0")
    u = _unit_xy(closing_axis_xy, name="closing_axis_xy")
    ex = _unit_xy(cube_x_axis_xy, name="cube_x_axis_xy")
    ey = _unit_xy(cube_y_axis_xy, name="cube_y_axis_xy")
    return float(cube_side_m) * (
        torch.sum(u * ex, dim=-1).abs() + torch.sum(u * ey, dim=-1).abs()
    )


def grasp_alignment_diagnostics(
    closing_axis_xy: Tensor,
    cube_x_axis_xy: Tensor,
    cube_y_axis_xy: Tensor,
    *,
    cube_side_m: float,
) -> GraspAlignmentDiagnostics:
    """Return edge-angle and effective cube width for one/batched grasp axes."""
    u = _unit_xy(closing_axis_xy, name="closing_axis_xy")
    ex = _unit_xy(cube_x_axis_xy, name="cube_x_axis_xy")
    ey = _unit_xy(cube_y_axis_xy, name="cube_y_axis_xy")
    angle_x = undirected_angle_deg(u, ex)
    angle_y = undirected_angle_deg(u, ey)
    nearest = torch.minimum(angle_x, angle_y)
    width = square_projected_width_m(u, ex, ey, cube_side_m=cube_side_m)
    return GraspAlignmentDiagnostics(
        closing_axis_xy=u,
        cube_x_axis_xy=ex,
        cube_y_axis_xy=ey,
        angle_to_cube_x_deg=angle_x,
        angle_to_cube_y_deg=angle_y,
        nearest_edge_angle_deg=nearest,
        projected_cube_width_m=width,
    )


def edge_alignment_command(
    closing_axis_xy: Tensor,
    cube_x_axis_xy: Tensor,
    cube_y_axis_xy: Tensor,
    *,
    max_step_rad: float,
    tolerance_deg: float,
) -> EdgeAlignmentCommand:
    """Compute the shortest signed yaw step that aligns closing axis to an edge.

    The gripper closing axis is undirected for grasp geometry: aligning to +X
    or -X is equivalent, as is +Y or -Y.  We therefore evaluate all four
    directed candidates ``(+X, -X, +Y, -Y)`` and choose the one requiring the
    smallest absolute yaw rotation.  Positive error is counter-clockwise in
    the XY plane.

    ``bounded_step_rad`` is zero inside ``tolerance_deg`` and otherwise clipped
    to ``[-max_step_rad, +max_step_rad]``.  The values are *processed* pose
    deltas in radians; conversion to the raw Isaac action scale belongs to the
    action adapter.
    """
    if max_step_rad <= 0:
        raise ValueError("max_step_rad must be > 0")
    if tolerance_deg < 0 or tolerance_deg >= 45.0:
        raise ValueError("tolerance_deg must satisfy 0 <= tolerance_deg < 45")

    u = _unit_xy(closing_axis_xy, name="closing_axis_xy")
    ex = _unit_xy(cube_x_axis_xy, name="cube_x_axis_xy")
    ey = _unit_xy(cube_y_axis_xy, name="cube_y_axis_xy")

    # Broadcast to (..., 4 candidates, 2).
    candidates = torch.stack((ex, -ex, ey, -ey), dim=-2)
    u4 = u.unsqueeze(-2).expand_as(candidates)
    errors = _signed_angle_rad(u4, candidates)
    best_idx = torch.argmin(errors.abs(), dim=-1)
    gather_idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(*best_idx.shape, 1, 2)
    target = torch.gather(candidates, dim=-2, index=gather_idx).squeeze(-2)
    best_error = torch.gather(errors, dim=-1, index=best_idx.unsqueeze(-1)).squeeze(-1)

    nearest_deg = torch.rad2deg(best_error.abs())
    step = best_error.clamp(min=-float(max_step_rad), max=float(max_step_rad))
    step = torch.where(nearest_deg <= float(tolerance_deg), torch.zeros_like(step), step)

    return EdgeAlignmentCommand(
        signed_error_rad=best_error,
        bounded_step_rad=step,
        target_axis_xy=target,
        target_axis_index=best_idx,
        nearest_edge_angle_deg=nearest_deg,
    )


def quat_wxyz_planar_axes(quat_wxyz: Tensor) -> tuple[Tensor, Tensor]:
    """Return projected local X/Y axes from a wxyz quaternion.

    Isaac Lab rigid-body root quaternions use wxyz ordering.  Only XY
    components are returned because this diagnostic concerns top-down grasp
    orientation.  Upright cubes yield orthogonal unit planar directions after
    normalization in ``grasp_alignment_diagnostics``.
    """
    q = torch.as_tensor(quat_wxyz)
    if q.ndim < 1 or q.shape[-1] != 4:
        raise ValueError(f"quat_wxyz must end in 4 values, got {tuple(q.shape)}")
    if not torch.isfinite(q).all():
        raise ValueError("quat_wxyz contains NaN/Inf")
    qn = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = qn.unbind(dim=-1)

    # First and second columns of the quaternion rotation matrix.
    ex = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
        ),
        dim=-1,
    )
    ey = torch.stack(
        (
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
        ),
        dim=-1,
    )
    return ex, ey
