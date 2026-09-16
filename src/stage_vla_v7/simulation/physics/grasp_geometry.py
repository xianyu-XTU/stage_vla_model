"""Geometry-conditioned parallel-jaw targets for the physical runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor


SUPPORTED_GEOMETRIES = ("box", "cylinder", "cone")


@dataclass(frozen=True)
class GraspGeometryProfile:
    geometry: str = "box"
    height_ratio: float = 0.5
    width_ratio: float | None = None

    def validate(self) -> None:
        if self.geometry not in SUPPORTED_GEOMETRIES:
            raise ValueError(f"unsupported grasp geometry: {self.geometry!r}")
        if not 0.0 < float(self.height_ratio) < 1.0:
            raise ValueError("height_ratio must lie in (0, 1)")
        if self.width_ratio is not None and not 0.0 < float(self.width_ratio) <= 1.0:
            raise ValueError("width_ratio must lie in (0, 1]")


@dataclass(frozen=True)
class ParallelJawCandidate:
    target_position_w: Tensor
    effective_width_m: Tensor
    height_ratio: float
    width_ratio: Tensor


def parallel_jaw_yaw_error(
    object_quat_xyzw: Tensor,
    size_xyz_m: Tensor,
    left_tip_w: Tensor,
    right_tip_w: Tensor,
) -> Tensor:
    quat = torch.as_tensor(object_quat_xyzw, dtype=torch.float32)
    size = torch.as_tensor(size_xyz_m, device=quat.device, dtype=quat.dtype)
    left = torch.as_tensor(left_tip_w, device=quat.device, dtype=quat.dtype)
    right = torch.as_tensor(right_tip_w, device=quat.device, dtype=quat.dtype)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("object_quat_xyzw must have shape [N,4]")
    if size.shape != (len(quat), 3):
        raise ValueError("size_xyz_m must have shape [N,3]")
    if left.shape != (len(quat), 3) or right.shape != left.shape:
        raise ValueError("fingertip positions must have shape [N,3]")
    if not all(torch.isfinite(value).all() for value in (quat, size, left, right)):
        raise ValueError("yaw inputs must be finite")
    if torch.any(size <= 0):
        raise ValueError("size_xyz_m must be positive")
    quat_norm = quat.norm(dim=-1)
    if torch.any(quat_norm < 1.0e-6):
        raise ValueError("object quaternions must be non-zero")
    quat = quat / quat_norm.unsqueeze(-1)
    x, y, z, w = quat.unbind(dim=-1)
    object_x_yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )
    jaw_delta = right[:, :2] - left[:, :2]
    if torch.any(jaw_delta.norm(dim=-1) < 1.0e-6):
        raise ValueError("fingertips must define a non-zero jaw axis")
    jaw_yaw = torch.atan2(jaw_delta[:, 1], jaw_delta[:, 0])
    error_x = (
        torch.remainder(object_x_yaw - jaw_yaw + torch.pi / 2.0, torch.pi)
        - torch.pi / 2.0
    )
    error_y = (
        torch.remainder(
            object_x_yaw + torch.pi / 2.0 - jaw_yaw + torch.pi / 2.0,
            torch.pi,
        )
        - torch.pi / 2.0
    )
    return torch.where(error_x.abs() <= error_y.abs(), error_x, error_y)


def profile_for_geometry(
    geometry: str,
    *,
    height_ratio: float | None = None,
) -> GraspGeometryProfile:
    if geometry == "cone":
        profile = GraspGeometryProfile(
            geometry, 0.25 if height_ratio is None else height_ratio
        )
    else:
        profile = GraspGeometryProfile(
            geometry, 0.5 if height_ratio is None else height_ratio
        )
    profile.validate()
    return profile


def profile_from_config(
    path: str | Path,
    *,
    geometry: str | None = None,
) -> GraspGeometryProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    obj = payload.get("object", {})
    selected = str(geometry or obj.get("geometry", "box"))
    grasp = obj.get("grasp", {})
    profile = profile_for_geometry(
        selected,
        height_ratio=float(grasp["height_ratio"]) if "height_ratio" in grasp else None,
    )
    if "width_ratio" in grasp:
        profile = GraspGeometryProfile(
            profile.geometry,
            profile.height_ratio,
            float(grasp["width_ratio"]),
        )
        profile.validate()
    return profile


def local_width_ratio(
    geometry: str,
    height_ratio: float | Tensor,
    *,
    explicit_width_ratio: float | None = None,
) -> Tensor:
    profile = GraspGeometryProfile(
        geometry,
        float(height_ratio) if not isinstance(height_ratio, Tensor) else 0.5,
        explicit_width_ratio,
    )
    profile.validate()
    ratio = torch.as_tensor(height_ratio, dtype=torch.float32)
    if explicit_width_ratio is not None:
        return torch.full_like(ratio, float(explicit_width_ratio))
    if geometry == "cone":
        return (1.0 - ratio).clamp_min(0.05)
    return torch.ones_like(ratio)


def build_parallel_jaw_candidate(
    object_position_w: Tensor,
    size_xyz_m: tuple[float, float, float] | Tensor,
    *,
    profile: GraspGeometryProfile,
) -> ParallelJawCandidate:
    profile.validate()
    position = torch.as_tensor(object_position_w, dtype=torch.float32)
    if position.ndim < 1 or position.shape[-1] != 3:
        raise ValueError("object_position_w must have shape (..., 3)")
    size = torch.as_tensor(size_xyz_m, dtype=position.dtype, device=position.device)
    if (
        size.shape not in ((3,), position.shape)
        or not torch.isfinite(size).all()
        or (size <= 0).any()
    ):
        raise ValueError("size_xyz_m must be [3] or match object_position_w")
    ratio = torch.as_tensor(
        profile.height_ratio, dtype=position.dtype, device=position.device
    )
    width_ratio = local_width_ratio(
        profile.geometry,
        ratio,
        explicit_width_ratio=profile.width_ratio,
    ).to(device=position.device, dtype=position.dtype)
    target = position.clone()
    height = size[2] if size.ndim == 1 else size[..., 2]
    width = size[0] if size.ndim == 1 else size[..., 0]
    target[..., 2] += (ratio - 0.5) * height
    effective_width = width * width_ratio
    return ParallelJawCandidate(
        target, effective_width, float(profile.height_ratio), width_ratio
    )


def grasp_target_position(
    object_position_w: Tensor,
    size_xyz_m: tuple[float, float, float] | Tensor,
    *,
    profile: GraspGeometryProfile,
) -> Tensor:
    return build_parallel_jaw_candidate(
        object_position_w, size_xyz_m, profile=profile
    ).target_position_w
