"""M7 pure-PyTorch red-on-blue placement diagnostics.

Official Isaac Lab stack semantics establish cube_2 (red) as the upper object
and cube_1 (blue) as the lower object for stack_1.

This project uses a stricter signed-height check than the generic official
``object_stacked`` helper:

    xy_error <= xy_tolerance
    AND red_z - blue_z > 0
    AND |(red_z - blue_z) - target_height_diff| <= height_tolerance

M7 additionally checks low linear/angular speed for both red and blue cubes.
Low-speed gating is a project design choice, not an Isaac Lab requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RedOnBlueConfig:
    xy_tolerance_m: float
    target_height_diff_m: float
    height_tolerance_m: float
    max_red_linear_speed_mps: float
    max_red_angular_speed_radps: float
    max_blue_linear_speed_mps: float
    max_blue_angular_speed_radps: float

    def validate(self) -> None:
        if self.xy_tolerance_m <= 0:
            raise ValueError("xy_tolerance_m must be > 0")
        if self.target_height_diff_m <= 0:
            raise ValueError("target_height_diff_m must be > 0")
        if self.height_tolerance_m < 0:
            raise ValueError("height_tolerance_m must be >= 0")
        for name, value in (
            ("max_red_linear_speed_mps", self.max_red_linear_speed_mps),
            ("max_red_angular_speed_radps", self.max_red_angular_speed_radps),
            ("max_blue_linear_speed_mps", self.max_blue_linear_speed_mps),
            ("max_blue_angular_speed_radps", self.max_blue_angular_speed_radps),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class RedOnBlueDiagnostics:
    geometry_ok: Tensor
    settled: Tensor
    xy_error_m: Tensor
    height_diff_m: Tensor
    height_error_m: Tensor
    red_above_blue: Tensor
    red_linear_speed_mps: Tensor
    red_angular_speed_radps: Tensor
    blue_linear_speed_mps: Tensor
    blue_angular_speed_radps: Tensor


def _check_xyz(name: str, value: Tensor) -> Tensor:
    value = torch.as_tensor(value)
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must end in xyz, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def red_on_blue_diagnostics(
    red_pos_w: Tensor,
    blue_pos_w: Tensor,
    red_lin_vel_w: Tensor,
    red_ang_vel_w: Tensor,
    blue_lin_vel_w: Tensor,
    blue_ang_vel_w: Tensor,
    *,
    cfg: RedOnBlueConfig,
) -> RedOnBlueDiagnostics:
    cfg.validate()

    red_pos = _check_xyz("red_pos_w", red_pos_w)
    blue_pos = _check_xyz("blue_pos_w", blue_pos_w)
    red_lin = _check_xyz("red_lin_vel_w", red_lin_vel_w)
    red_ang = _check_xyz("red_ang_vel_w", red_ang_vel_w)
    blue_lin = _check_xyz("blue_lin_vel_w", blue_lin_vel_w)
    blue_ang = _check_xyz("blue_ang_vel_w", blue_ang_vel_w)

    if red_pos.shape != blue_pos.shape:
        raise ValueError("red_pos_w and blue_pos_w must share shape")
    for name, value in (
        ("red_lin_vel_w", red_lin),
        ("red_ang_vel_w", red_ang),
        ("blue_lin_vel_w", blue_lin),
        ("blue_ang_vel_w", blue_ang),
    ):
        if value.shape != red_pos.shape:
            raise ValueError(f"{name} must match position shape")

    delta = red_pos - blue_pos
    xy_error = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    height_diff = delta[..., 2]
    height_error = torch.abs(height_diff - cfg.target_height_diff_m)
    red_above = height_diff > 0.0

    geometry = (
        (xy_error <= cfg.xy_tolerance_m)
        & red_above
        & (height_error <= cfg.height_tolerance_m)
    )

    red_lin_speed = torch.linalg.vector_norm(red_lin, dim=-1)
    red_ang_speed = torch.linalg.vector_norm(red_ang, dim=-1)
    blue_lin_speed = torch.linalg.vector_norm(blue_lin, dim=-1)
    blue_ang_speed = torch.linalg.vector_norm(blue_ang, dim=-1)

    settled = (
        (red_lin_speed <= cfg.max_red_linear_speed_mps)
        & (red_ang_speed <= cfg.max_red_angular_speed_radps)
        & (blue_lin_speed <= cfg.max_blue_linear_speed_mps)
        & (blue_ang_speed <= cfg.max_blue_angular_speed_radps)
    )

    return RedOnBlueDiagnostics(
        geometry_ok=geometry,
        settled=settled,
        xy_error_m=xy_error,
        height_diff_m=height_diff,
        height_error_m=height_error,
        red_above_blue=red_above,
        red_linear_speed_mps=red_lin_speed,
        red_angular_speed_radps=red_ang_speed,
        blue_linear_speed_mps=blue_lin_speed,
        blue_angular_speed_radps=blue_ang_speed,
    )



def vertical_surface_clearance_m(
    upper_center_z_w: Tensor,
    lower_center_z_w: Tensor,
    *,
    upper_height_m: float,
    lower_height_m: float,
) -> Tensor:
    """Vertical gap from lower top face to upper bottom face.

    Positive:
        the upper object is physically above the lower object's top surface.
    Zero:
        ideal face-to-face contact for axis-aligned boxes.
    Negative:
        their vertical extents overlap.

    This is a geometric helper. It does not infer contact or stability.
    """
    if upper_height_m <= 0 or lower_height_m <= 0:
        raise ValueError("object heights must be > 0")

    upper_z = torch.as_tensor(upper_center_z_w)
    lower_z = torch.as_tensor(
        lower_center_z_w,
        device=upper_z.device,
        dtype=upper_z.dtype,
    )
    if upper_z.shape != lower_z.shape:
        raise ValueError(
            f"upper/lower center z must share shape; got "
            f"{tuple(upper_z.shape)} vs {tuple(lower_z.shape)}"
        )
    if not torch.isfinite(upper_z).all() or not torch.isfinite(lower_z).all():
        raise ValueError("center z contains NaN/Inf")

    upper_bottom = upper_z - 0.5 * upper_height_m
    lower_top = lower_z + 0.5 * lower_height_m
    return upper_bottom - lower_top


def safe_transport_upper_center_z(
    lower_center_z_w: Tensor,
    *,
    upper_height_m: float,
    lower_height_m: float,
    surface_clearance_m: float,
) -> Tensor:
    """Target upper-object center z that gives the requested face clearance."""
    if upper_height_m <= 0 or lower_height_m <= 0:
        raise ValueError("object heights must be > 0")
    if surface_clearance_m < 0:
        raise ValueError("surface_clearance_m must be >= 0")

    lower_z = torch.as_tensor(lower_center_z_w)
    if not torch.isfinite(lower_z).all():
        raise ValueError("lower_center_z_w contains NaN/Inf")

    return (
        lower_z
        + 0.5 * lower_height_m
        + surface_clearance_m
        + 0.5 * upper_height_m
    )
