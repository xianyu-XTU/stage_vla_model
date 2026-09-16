"""Small RGB-D object-state adapter for the v5 vision boundary.

The first implementation is deliberately model-independent: it is a compact
colour/depth baseline for the red/blue cube simulator scene.  It emits the
same :class:`~stage_vla.vision.state.Detection` objects expected by the task
adapter, so a learned detector can replace it without changing the scheduler
or skill policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..objects import OBJECT_CATALOG
from .state import Detection


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics and a camera-to-root homogeneous transform.

    The mapping follows the standard RGB-D deprojection used by OpenCV and
    Intel RealSense: a depth pixel ``(u, v, Z)`` is first back-projected to
    camera coordinates with ``X=(u-cx)Z/fx`` and ``Y=(v-cy)Z/fy`` and is then
    transformed with the calibrated rigid transform ``R @ p + t``.  The
    camera pose is deliberately supplied as data rather than inferred from
    the object detector, so calibration errors remain visible and auditable.
    References: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html and
    https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20.
    """

    intrinsic: np.ndarray
    camera_to_root: np.ndarray

    def validate(self) -> None:
        k = np.asarray(self.intrinsic, dtype=np.float64)
        t = np.asarray(self.camera_to_root, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"intrinsic must be [3,3], got {k.shape}")
        if t.shape != (4, 4):
            raise ValueError(f"camera_to_root must be [4,4], got {t.shape}")
        if not np.isfinite(k).all() or not np.isfinite(t).all():
            raise ValueError("camera calibration contains NaN/Inf")
        if abs(float(k[2, 2])) < 1e-9:
            raise ValueError("intrinsic[2,2] must be non-zero")

    def deproject_pixel(self, uv: np.ndarray, depth_m: float) -> np.ndarray:
        """Map one image pixel and metric depth to the robot-root frame."""
        self.validate()
        pixel = np.asarray(uv, dtype=np.float64).reshape(-1)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("uv must contain two finite pixel coordinates")
        depth = float(depth_m)
        k = np.asarray(self.intrinsic, dtype=np.float64)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        if fx <= 0.0 or fy <= 0.0 or not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("invalid pinhole intrinsics or depth")
        camera = np.array([
            (pixel[0] - cx) * depth / fx,
            (pixel[1] - cy) * depth / fy,
            depth,
            1.0,
        ], dtype=np.float64)
        root = np.asarray(self.camera_to_root, dtype=np.float64) @ camera
        if abs(float(root[3])) > 1e-9:
            root = root / root[3]
        return root[:3].astype(np.float32)

    def deproject_pixels(self, uv: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Vectorized version of :meth:`deproject_pixel` for ``[...,2]`` pixels."""
        self.validate()
        pixels = np.asarray(uv, dtype=np.float64)
        depths = np.asarray(depth_m, dtype=np.float64)
        if pixels.shape[-1:] != (2,) or pixels.ndim == 0:
            raise ValueError("uv must have shape [...,2]")
        if depths.shape != pixels.shape[:-1]:
            raise ValueError("depth_m shape must match uv without its last axis")
        if not np.isfinite(pixels).all() or not np.isfinite(depths).all() or (depths <= 0.0).any():
            raise ValueError("uv and depth_m must be finite and depth_m must be positive")
        k = np.asarray(self.intrinsic, dtype=np.float64)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("invalid pinhole intrinsics")
        camera = np.empty((*depths.shape, 4), dtype=np.float64)
        camera[..., 0] = (pixels[..., 0] - cx) * depths / fx
        camera[..., 1] = (pixels[..., 1] - cy) * depths / fy
        camera[..., 2] = depths
        camera[..., 3] = 1.0
        root = camera @ np.asarray(self.camera_to_root, dtype=np.float64).T
        w = root[..., 3:4]
        root = np.divide(root, w, out=root, where=np.abs(w) > 1e-9)
        return root[..., :3].astype(np.float32)


def _mask_for_label(rgb: np.ndarray, label: str) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"rgb must have shape [H,W,3+], got {image.shape}")
    image = image[..., :3].astype(np.float32, copy=False)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    if label == "red_cube":
        return (r >= 60.0) & (r > 1.20 * g) & (r > 1.20 * b) & ((r - g) >= 20.0)
    if label == "blue_cube":
        return (b >= 45.0) & (b > 1.15 * r) & (b > 1.15 * g) & ((b - r) >= 12.0)
    if label == "green_cube":
        return (g >= 45.0) & (g > 1.15 * r) & (g > 1.15 * b) & ((g - r) >= 12.0)
    if label == "yellow_cube":
        return (r >= 80.0) & (g >= 70.0) & (b <= 110.0) & (r > 1.25 * b) & (g > 1.15 * b)
    # New rigid/deformable objects use catalog color metadata until a learned
    # detector is installed.  The broad distance gate is intentionally
    # conservative and remains replaceable at the vision boundary.
    spec = OBJECT_CATALOG.get(label)
    if spec is None or spec.color_rgb is None:
        raise ValueError(f"unsupported compact colour label: {label!r}")
    reference = np.asarray(spec.color_rgb, dtype=np.float32).reshape(1, 1, 3)
    distance = np.linalg.norm(image - reference, axis=-1)
    brightness = image.mean(axis=-1)
    return (distance <= 55.0) & (brightness >= 25.0)


def _pca_yaw(points_uv: np.ndarray) -> float:
    if len(points_uv) < 3:
        return 0.0
    centered = points_uv.astype(np.float64) - points_uv.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points_uv) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(np.argmax(values))]
    return float(math.atan2(direction[1], direction[0]))


def _backproject(uv: np.ndarray, depth: float, calibration: CameraCalibration) -> np.ndarray:
    return calibration.deproject_pixel(uv, depth)


class CompactColorDepthDetector:
    """Detect catalog objects from one RGB-D frame.

    Cube labels retain their calibrated color thresholds; other catalog
    objects use the generic metadata-driven fallback until a learned model is
    available.
    """

    def __init__(
        self,
        *,
        min_pixels: int = 12,
        confidence_floor: float = 0.20,
        position_bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        if min_pixels < 1:
            raise ValueError("min_pixels must be positive")
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0,1]")
        bias = np.asarray(position_bias_m, dtype=np.float64).reshape(-1)
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("position_bias_m must be finite xyz")
        self.min_pixels = int(min_pixels)
        self.confidence_floor = float(confidence_floor)
        self.position_bias_m = bias.astype(np.float32)

    def detect_one(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        label: str,
        calibration: CameraCalibration,
    ) -> Detection | None:
        calibration.validate()
        image = np.asarray(rgb)
        depth = np.asarray(depth_m, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"rgb must have shape [H,W,3+], got {image.shape}")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != image.shape[:2]:
            raise ValueError(f"depth shape {depth.shape} does not match RGB {image.shape[:2]}")
        mask = _mask_for_label(image, label)
        valid = mask & np.isfinite(depth) & (depth > 0.05)
        ys, xs = np.nonzero(valid)
        if len(xs) < self.min_pixels:
            return None
        uv = np.stack([xs, ys], axis=1).astype(np.float64)
        # Median depth suppresses edge pixels and isolated rendering artefacts.
        z = float(np.median(depth[ys, xs]))
        center_uv = np.median(uv, axis=0)
        position = _backproject(center_uv, z, calibration) + self.position_bias_m
        area = float(len(xs))
        image_area = float(mask.shape[0] * mask.shape[1])
        fill = min(area / max(image_area * 0.002, 1.0), 1.0)
        valid_ratio = min(float(len(xs)) / max(float(mask.sum()), 1.0), 1.0)
        confidence = float(np.clip(0.15 + 0.55 * fill + 0.30 * valid_ratio, 0.0, 1.0))
        if confidence < self.confidence_floor:
            return None
        return Detection(
            label=label,
            position_xyz_m=tuple(float(v) for v in position),
            yaw_rad=_pca_yaw(uv),
            confidence=confidence,
        )

    def detect_scene(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        calibration: CameraCalibration,
        labels: tuple[str, ...] = ("red_cube", "blue_cube"),
    ) -> tuple[Detection, ...]:
        if not labels:
            raise ValueError("labels must not be empty")
        detections = []
        for label in labels:
            detection = self.detect_one(rgb, depth_m, label=label, calibration=calibration)
            if detection is not None:
                detections.append(detection)
        return tuple(detections)


def infer_held_label(
    detections: tuple[Detection, ...] | list[Detection],
    gripper_midpoint_xyz_m: tuple[float, float, float] | np.ndarray,
    *,
    max_distance_m: float = 0.075,
) -> str | None:
    """Use a bounded gripper-proximity cue to expose held-object state.

    This is intentionally a separate, auditable cue rather than a hidden
    scheduler rule.  A learned visual temporal cue can replace it later.
    """
    if max_distance_m <= 0.0:
        raise ValueError("max_distance_m must be positive")
    midpoint = np.asarray(gripper_midpoint_xyz_m, dtype=np.float64).reshape(-1)
    if midpoint.shape != (3,) or not np.isfinite(midpoint).all():
        raise ValueError("gripper_midpoint_xyz_m must be finite xyz")
    candidates = []
    for detection in detections:
        detection.validate()
        distance = float(np.linalg.norm(np.asarray(detection.position_xyz_m) - midpoint))
        if distance <= max_distance_m:
            candidates.append((distance, detection.label))
    return min(candidates)[1] if candidates else None
