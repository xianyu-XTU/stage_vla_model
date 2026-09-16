"""Optional YOLO instance-segmentation adapter for the v5 RGB-D boundary."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .rgbd_detector import CameraCalibration, CompactColorDepthDetector, _pca_yaw
from .state import Detection


class YoloCubeRGBDDetector:
    """Convert YOLO cube masks and metric depth into robot-root detections.

    Ultralytics remains an optional pilot dependency: importing ``stage_vla``
    does not require it, while constructing this adapter does.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | int = "0",
        confidence_floor: float = 0.25,
        imgsz: int = 256,
        labels: tuple[str, ...] = ("red_cube", "blue_cube"),
        position_bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0,1]")
        if imgsz < 32:
            raise ValueError("imgsz must be at least 32")
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("labels must be distinct and non-empty")
        bias = np.asarray(position_bias_m, dtype=np.float64).reshape(-1)
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("position_bias_m must be finite xyz")
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint_path))
        self.checkpoint = checkpoint_path
        self.device = (
            str(device).split(":", 1)[1]
            if isinstance(device, str) and device.startswith("cuda:")
            else device
        )
        self.confidence_floor = float(confidence_floor)
        self.imgsz = int(imgsz)
        self.labels = tuple(labels)
        self.position_bias_m = bias.astype(np.float32)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, ids, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(mask, dtype=np.uint8), connectivity=8
        )
        if count <= 1:
            return np.zeros_like(mask, dtype=bool)
        component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        return ids == component

    def _decode_result(
        self,
        result,
        depth_m: np.ndarray,
        calibration: CameraCalibration,
    ) -> tuple[Detection, ...]:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"depth must have shape [H,W] or [H,W,1], got {depth.shape}")
        candidates: dict[str, list[tuple[float, np.ndarray]]] = {
            label: [] for label in self.labels
        }
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidence = result.boxes.conf.detach().cpu().numpy()
            masks = result.masks.data.detach().cpu().numpy()
            for class_id, score, raw_mask in zip(classes, confidence, masks):
                label = result.names.get(int(class_id))
                if label not in candidates or float(score) < self.confidence_floor:
                    continue
                if raw_mask.shape != depth.shape:
                    raw_mask = cv2.resize(
                        raw_mask.astype(np.float32),
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                candidates[label].append((float(score), self._largest_component(raw_mask >= 0.5)))

        detections = []
        for label in self.labels:
            options = sorted(candidates[label], key=lambda item: item[0], reverse=True)
            if not options:
                continue
            score, mask = options[0]
            valid = mask & np.isfinite(depth) & (depth > 0.05)
            ys, xs = np.nonzero(valid)
            if len(xs) < 12:
                continue
            uv = np.stack([xs, ys], axis=1).astype(np.float64)
            center_uv = np.median(uv, axis=0)
            center_depth = float(np.median(depth[ys, xs]))
            position = calibration.deproject_pixel(center_uv, center_depth) + self.position_bias_m
            detection = Detection(
                label=label,
                position_xyz_m=tuple(float(value) for value in position),
                yaw_rad=_pca_yaw(uv),
                confidence=score,
            )
            detection.validate()
            detections.append(detection)
        return tuple(detections)

    def detect_batch(
        self,
        images,
        depth_m,
        *,
        calibration: CameraCalibration,
    ) -> tuple[tuple[Detection, ...], ...]:
        rgb = np.asarray(images)
        depth = np.asarray(depth_m)
        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise ValueError(f"images must have shape [N,H,W,3+], got {rgb.shape}")
        if depth.shape[0] != rgb.shape[0]:
            raise ValueError("RGB and depth batch sizes differ")
        sources = [cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR) for frame in rgb]
        results = self.model.predict(
            source=sources,
            imgsz=self.imgsz,
            conf=self.confidence_floor,
            device=self.device,
            batch=len(sources),
            retina_masks=True,
            verbose=False,
        )
        return tuple(
            self._decode_result(result, depth[index], calibration)
            for index, result in enumerate(results)
        )

    def detect_one(
        self,
        rgb,
        depth_m,
        *,
        calibration: CameraCalibration,
    ) -> tuple[Detection, ...]:
        return self.detect_batch(
            np.asarray(rgb)[None], np.asarray(depth_m)[None], calibration=calibration
        )[0]

    def detect_one_hybrid(
        self,
        rgb,
        depth_m,
        *,
        calibration: CameraCalibration,
        fallback: CompactColorDepthDetector | None = None,
        disagreement_gate_m: float = 0.05,
    ) -> tuple[tuple[Detection, ...], bool]:
        if disagreement_gate_m <= 0.0:
            raise ValueError("disagreement_gate_m must be positive")
        learned = self.detect_one(rgb, depth_m, calibration=calibration)
        learned_by_label = {item.label: item for item in learned}
        geometric = (fallback or CompactColorDepthDetector()).detect_scene(
            rgb, depth_m, calibration=calibration, labels=self.labels
        )
        geometric_by_label = {item.label: item for item in geometric}
        use_fallback = set(learned_by_label) != set(self.labels)
        if not use_fallback and set(geometric_by_label) == set(self.labels):
            use_fallback = any(
                float(
                    np.linalg.norm(
                        np.asarray(learned_by_label[label].position_xyz_m[:2])
                        - np.asarray(geometric_by_label[label].position_xyz_m[:2])
                    )
                ) > disagreement_gate_m
                for label in self.labels
            )
        if use_fallback and set(geometric_by_label) == set(self.labels):
            return tuple(geometric_by_label[label] for label in self.labels), True
        return learned, False
