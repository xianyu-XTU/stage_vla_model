"""Native color-segmentation and metric-depth provider for the cube domain."""

from __future__ import annotations

import math
from typing import Any

from stage_vla_v7.interfaces import (
    ModelDescriptor,
    ObjectDetection,
    ProviderError,
    SceneState,
    VisionRequest,
    VisionResult,
)

from ..geometry import CameraCalibration


RUNTIME_CUBE_LABELS = ("red_cube", "blue_cube", "green_cube", "yellow_cube")


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - RGB-D providers require numpy
        raise RuntimeError("compact color-depth detection requires numpy") from exc
    return numpy


def _mask_for_label(rgb: object, label: str) -> Any:
    np = _numpy()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"rgb must have shape [H,W,3+], got {image.shape}")
    image = image[..., :3].astype(np.float32, copy=False)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    if label == "red_cube":
        return (
            (red >= 60.0)
            & (red > 1.20 * green)
            & (red > 1.20 * blue)
            & ((red - green) >= 20.0)
        )
    if label == "blue_cube":
        return (
            (blue >= 45.0)
            & (blue > 1.15 * red)
            & (blue > 1.15 * green)
            & ((blue - red) >= 12.0)
        )
    if label == "green_cube":
        return (
            (green >= 45.0)
            & (green > 1.15 * red)
            & (green > 1.15 * blue)
            & ((green - red) >= 12.0)
        )
    if label == "yellow_cube":
        return (
            (red >= 80.0)
            & (green >= 70.0)
            & (blue <= 110.0)
            & (red > 1.25 * blue)
            & (green > 1.15 * blue)
        )
    raise ValueError(f"unsupported V7 cube colour label: {label!r}")


def _pca_yaw(points_uv: Any) -> float:
    np = _numpy()
    if len(points_uv) < 3:
        return 0.0
    centered = points_uv.astype(np.float64) - points_uv.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points_uv) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    return float(math.atan2(direction[1], direction[0]))


class CompactColorDepthDetector:
    """Estimate poses for the four cube labels supported by the current runtime."""

    def __init__(
        self,
        *,
        min_pixels: int = 12,
        confidence_floor: float = 0.20,
        position_bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        np = _numpy()
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
        rgb: object,
        depth_m: object,
        *,
        label: str,
        calibration: CameraCalibration,
    ) -> ObjectDetection | None:
        np = _numpy()
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
        rows, columns = np.nonzero(valid)
        if len(columns) < self.min_pixels:
            return None
        pixels = np.stack([columns, rows], axis=1).astype(np.float64)
        median_depth = float(np.median(depth[rows, columns]))
        center_uv = np.median(pixels, axis=0)
        position = (
            calibration.deproject_pixel(center_uv, median_depth) + self.position_bias_m
        )
        area = float(len(columns))
        image_area = float(mask.shape[0] * mask.shape[1])
        fill = min(area / max(image_area * 0.002, 1.0), 1.0)
        valid_ratio = min(float(len(columns)) / max(float(mask.sum()), 1.0), 1.0)
        confidence = float(
            np.clip(0.15 + 0.55 * fill + 0.30 * valid_ratio, 0.0, 1.0)
        )
        if confidence < self.confidence_floor:
            return None
        return ObjectDetection(
            label=label,
            position_xyz_m=tuple(float(value) for value in position),
            yaw_rad=_pca_yaw(pixels),
            confidence=confidence,
        )

    def detect_scene(
        self,
        rgb: object,
        depth_m: object,
        *,
        calibration: CameraCalibration,
        labels: tuple[str, ...] = ("red_cube", "blue_cube"),
    ) -> tuple[ObjectDetection, ...]:
        if not labels:
            raise ValueError("labels must not be empty")
        detections = []
        for label in labels:
            detection = self.detect_one(
                rgb,
                depth_m,
                label=label,
                calibration=calibration,
            )
            if detection is not None:
                detections.append(detection)
        return tuple(detections)


class CompactColorDepthProvider:
    """Expose the native detector through the public V7 VisionProvider contract."""

    descriptor = ModelDescriptor(
        name="v7-compact-color-depth",
        version="1",
        kind="vision",
        capabilities=("native-v7", "rgbd", "cube-color-segmentation"),
    )

    def __init__(
        self,
        *,
        calibration: CameraCalibration,
        labels: tuple[str, ...] = ("red_cube", "blue_cube"),
        detector: CompactColorDepthDetector | None = None,
    ) -> None:
        if not isinstance(calibration, CameraCalibration):
            raise TypeError("calibration must be CameraCalibration")
        calibration.validate()
        if not labels:
            raise ValueError("labels must not be empty")
        unsupported = [label for label in labels if label not in RUNTIME_CUBE_LABELS]
        if unsupported:
            raise ValueError(f"unsupported V7 cube labels: {unsupported}")
        self.calibration = calibration
        self.labels = tuple(labels)
        self.detector = detector or CompactColorDepthDetector()

    def detect(self, request: VisionRequest) -> VisionResult:
        if request.depth_m is None:
            raise ProviderError("compact color-depth detection requires metric depth")
        detections = self.detector.detect_scene(
            request.rgb,
            request.depth_m,
            calibration=self.calibration,
            labels=self.labels,
        )
        scene = SceneState(
            detections,
            frame_id=request.frame_id,
            timestamp_s=request.timestamp_s,
            held_label=request.metadata.get("held_label"),
        )
        return VisionResult(
            scene,
            self.descriptor,
            {"used_fallback": False, "backend": "compact_color_depth"},
        )
