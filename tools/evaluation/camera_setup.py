"""Dependency-light camera declarations and calibration transforms."""

from __future__ import annotations

import numpy as np

from . import bootstrap as _bootstrap  # noqa: F401
from stage_vla_v7.simulation.config import CameraSpec
from stage_vla_v7.simulation.isaac_lab import OBSERVER_CAMERA_NAME, VISION_CAMERA_NAME
from stage_vla_v7.simulation.models import ObserverCameraModel


def build_evaluation_camera_specs(
    *,
    use_vision: bool,
    observer: ObserverCameraModel | None,
    video_width: int | None = None,
    video_height: int | None = None,
    video_fps: float | None = None,
) -> tuple[CameraSpec, ...]:
    """Build independent model-input and observer camera declarations."""
    specs: list[CameraSpec] = []
    if use_vision:
        specs.append(CameraSpec(
            name=VISION_CAMERA_NAME,
            role="vision",
            width=128,
            height=128,
            data_types=("rgb", "distance_to_image_plane"),
        ))
    if observer is not None:
        specs.append(observer.to_camera_spec(
            width=video_width,
            height=video_height,
            fps=video_fps,
        ))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Vision and Observer camera IDs must be distinct")
    return tuple(specs)


def camera_cfg_transform(
    quat: np.ndarray,
    position: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Return the camera-to-environment-root transform from an Isaac ROS pose."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("camera quaternion must be finite xyzw")
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64) - np.asarray(
        origin, dtype=np.float64
    )
    return transform


__all__ = ["build_evaluation_camera_specs", "camera_cfg_transform"]
