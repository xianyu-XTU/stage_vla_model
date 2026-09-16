"""Validated pinhole calibration and camera-to-root projection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - RGB-D providers require numpy
        raise RuntimeError("camera calibration requires numpy") from exc
    return numpy


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics and a homogeneous camera-to-root transform."""

    intrinsic: object
    camera_to_root: object

    def validate(self) -> None:
        np = _numpy()
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        transform = np.asarray(self.camera_to_root, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be [3,3], got {intrinsic.shape}")
        if transform.shape != (4, 4):
            raise ValueError(f"camera_to_root must be [4,4], got {transform.shape}")
        if not np.isfinite(intrinsic).all() or not np.isfinite(transform).all():
            raise ValueError("camera calibration contains NaN/Inf")
        if abs(float(intrinsic[2, 2])) < 1e-9:
            raise ValueError("intrinsic[2,2] must be non-zero")

    def deproject_pixel(self, uv: object, depth_m: float) -> Any:
        """Map one image pixel and metric depth to robot-root coordinates."""
        np = _numpy()
        self.validate()
        pixel = np.asarray(uv, dtype=np.float64).reshape(-1)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("uv must contain two finite pixel coordinates")
        depth = float(depth_m)
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        if fx <= 0.0 or fy <= 0.0 or not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("invalid pinhole intrinsics or depth")
        camera = np.array(
            [
                (pixel[0] - cx) * depth / fx,
                (pixel[1] - cy) * depth / fy,
                depth,
                1.0,
            ],
            dtype=np.float64,
        )
        root = np.asarray(self.camera_to_root, dtype=np.float64) @ camera
        if abs(float(root[3])) > 1e-9:
            root = root / root[3]
        return root[:3].astype(np.float32)

    def deproject_pixels(self, uv: object, depth_m: object) -> Any:
        """Vectorized projection for pixels with shape ``[...,2]``."""
        np = _numpy()
        self.validate()
        pixels = np.asarray(uv, dtype=np.float64)
        depths = np.asarray(depth_m, dtype=np.float64)
        if pixels.shape[-1:] != (2,) or pixels.ndim == 0:
            raise ValueError("uv must have shape [...,2]")
        if depths.shape != pixels.shape[:-1]:
            raise ValueError("depth_m shape must match uv without its last axis")
        if (
            not np.isfinite(pixels).all()
            or not np.isfinite(depths).all()
            or (depths <= 0.0).any()
        ):
            raise ValueError("uv and depth_m must be finite and depth_m must be positive")
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("invalid pinhole intrinsics")
        camera = np.empty((*depths.shape, 4), dtype=np.float64)
        camera[..., 0] = (pixels[..., 0] - cx) * depths / fx
        camera[..., 1] = (pixels[..., 1] - cy) * depths / fy
        camera[..., 2] = depths
        camera[..., 3] = 1.0
        root = camera @ np.asarray(self.camera_to_root, dtype=np.float64).T
        weights = root[..., 3:4]
        root = np.divide(root, weights, out=root, where=np.abs(weights) > 1e-9)
        return root[..., :3].astype(np.float32)
