"""Tiny RGB-D detector used after the camera-contract baseline.

The network predicts red/blue cube centers in the robot-root frame and one
confidence logit per object.  It is intentionally independent of the action
policies and can be replaced without changing :mod:`stage_vla.vision.state`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .rgbd_detector import CameraCalibration, CompactColorDepthDetector
from .state import Detection, SceneState, VisionStateAdapter


class CompactRGBDDetectorNet(nn.Module):
    """Small spatial CNN: 128x128x4 RGB-D -> 2x(3-D position + confidence)."""

    output_dim = 8

    def __init__(self, hidden: int = 128):
        super().__init__()
        if hidden < 16:
            raise ValueError("hidden must be >= 16")
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 24, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * 8 * 8, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.output_dim),
        )

    def forward(self, image_rgbd: torch.Tensor) -> torch.Tensor:
        if image_rgbd.ndim != 4 or image_rgbd.shape[1] != 4:
            raise ValueError("expected [N,4,H,W] RGB-D input")
        return self.head(self.backbone(image_rgbd))


def pack_rgbd(images, depth_m, *, device=None) -> torch.Tensor:
    """Convert uint8 HWC RGB and metre depth arrays to normalized NCHW input."""
    rgb = torch.as_tensor(images, dtype=torch.float32, device=device)
    depth = torch.as_tensor(depth_m, dtype=torch.float32, device=device)
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError(f"images must be [N,H,W,3+], got {tuple(rgb.shape)}")
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    if depth.shape[:3] != rgb.shape[:3] or depth.shape[-1] != 1:
        raise ValueError(f"depth shape {tuple(depth.shape)} incompatible with RGB {tuple(rgb.shape)}")
    rgb = rgb[..., :3] / 255.0
    depth = depth.clamp(0.0, 4.0) / 4.0
    return torch.cat([rgb, depth], dim=-1).permute(0, 3, 1, 2).contiguous()


def decode_positions(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode network output into ``positions[N,2,3]`` and confidence[N,2]."""
    if raw.ndim != 2 or raw.shape[-1] != CompactRGBDDetectorNet.output_dim:
        raise ValueError(f"expected [N,8] detector output, got {tuple(raw.shape)}")
    positions = raw[:, :6].reshape(-1, 2, 3)
    confidence = raw[:, 6:8].sigmoid()
    return positions, confidence


class LearnedRGBDDetector:
    """Run a scripted compact detector and emit validated vision-state objects."""

    labels = ("red_cube", "blue_cube")

    def __init__(
        self,
        checkpoint: str | Path,
        normalization: str | Path,
        *,
        device: str | torch.device = "cpu",
        confidence_floor: float = 0.50,
        labels: tuple[str, str] | None = None,
    ):
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0,1]")
        checkpoint_path = Path(checkpoint).resolve()
        normalization_path = Path(normalization).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not normalization_path.is_file():
            raise FileNotFoundError(normalization_path)
        self.device = torch.device(device)
        self.confidence_floor = float(confidence_floor)
        if labels is not None:
            if len(labels) != 2 or len(set(labels)) != 2:
                raise ValueError("the current checkpoint supports exactly two distinct labels")
            self.labels = tuple(labels)
        self.model = torch.jit.load(str(checkpoint_path), map_location=self.device).eval()
        payload = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.position_mean = torch.as_tensor(
            payload["position_mean"], dtype=torch.float32, device=self.device
        )
        self.position_std = torch.as_tensor(
            payload["position_std"], dtype=torch.float32, device=self.device
        )
        if self.position_mean.shape != (3,) or self.position_std.shape != (3,):
            raise ValueError("normalization must contain three position means and stds")
        if not torch.isfinite(self.position_mean).all() or not torch.isfinite(self.position_std).all():
            raise ValueError("normalization contains NaN/Inf")
        if torch.any(self.position_std <= 0):
            raise ValueError("position std must be positive")

    def predict_batch(self, images, depth_m) -> tuple[np.ndarray, np.ndarray]:
        """Predict object positions [m] and confidences for an RGB-D batch."""
        with torch.inference_mode():
            raw = self.model(pack_rgbd(images, depth_m, device=self.device))
            normalized, confidence = decode_positions(raw)
            positions = normalized * self.position_std + self.position_mean
        return positions.cpu().numpy(), confidence.cpu().numpy()

    def detect_one(self, rgb, depth_m) -> tuple[Detection, ...]:
        """Return confidence-gated detections for one RGB-D frame."""
        image = np.asarray(rgb)
        depth = np.asarray(depth_m)
        if image.ndim != 3:
            raise ValueError(f"rgb must be one HWC frame, got {image.shape}")
        if depth.ndim not in (2, 3):
            raise ValueError(f"depth must be one HW or HW1 frame, got {depth.shape}")
        positions, confidence = self.predict_batch(image[None], depth[None])
        detections = []
        for index, label in enumerate(self.labels):
            score = float(confidence[0, index])
            if score < self.confidence_floor:
                continue
            detection = Detection(
                label=label,
                position_xyz_m=tuple(float(value) for value in positions[0, index]),
                yaw_rad=0.0,
                confidence=score,
            )
            detection.validate()
            detections.append(detection)
        return tuple(detections)

    def detect_one_hybrid(
        self,
        rgb,
        depth_m,
        *,
        calibration: CameraCalibration,
        fallback: CompactColorDepthDetector | None = None,
        disagreement_gate_m: float = 0.05,
    ) -> tuple[tuple[Detection, ...], bool]:
        """Use the learned output, falling back to RGB-D geometry on disagreement.

        The fallback is explicit and auditable: it is used only when the
        learned model misses an object or disagrees with the calibrated
        RGB-D centroid by more than ``disagreement_gate_m``.
        """
        if disagreement_gate_m <= 0:
            raise ValueError("disagreement_gate_m must be positive")
        learned = self.detect_one(rgb, depth_m)
        learned_by_label = {item.label: item for item in learned}
        geometric = (fallback or CompactColorDepthDetector()).detect_scene(
            rgb, depth_m, calibration=calibration
        )
        geometric_by_label = {item.label: item for item in geometric}
        use_fallback = set(learned_by_label) != set(self.labels)
        if not use_fallback:
            for label in self.labels:
                delta = np.linalg.norm(
                    np.asarray(learned_by_label[label].position_xyz_m)[:2]
                    - np.asarray(geometric_by_label.get(label, learned_by_label[label]).position_xyz_m)[:2]
                )
                if delta > disagreement_gate_m:
                    use_fallback = True
                    break
        if use_fallback and set(geometric_by_label) == set(self.labels):
            reconciled = []
            for label in self.labels:
                geometry = geometric_by_label[label]
                learned_item = learned_by_label.get(label)
                z = geometry.position_xyz_m[2] if learned_item is None else learned_item.position_xyz_m[2]
                reconciled.append(Detection(
                    label=label,
                    position_xyz_m=(geometry.position_xyz_m[0], geometry.position_xyz_m[1], z),
                    yaw_rad=0.0,
                    confidence=geometry.confidence if learned_item is None else learned_item.confidence,
                ))
            return tuple(reconciled), True
        return learned, False

    def detect_one_fused(
        self,
        rgb,
        depth_m,
        *,
        calibration: CameraCalibration,
        fallback: CompactColorDepthDetector | None = None,
        geometry_weight: float = 0.5,
        disagreement_gate_m: float = 0.05,
    ) -> tuple[tuple[Detection, ...], bool]:
        """Fuse learned and calibrated RGB-D XY estimates when both exist.

        The compact network and the colour/depth centroid have complementary
        failure modes near the gripper.  A bounded convex blend avoids the
        discontinuity of hard fallback while retaining the explicit fallback
        path when either estimator is unavailable.  Z is kept from the learned
        estimate because depth-to-root calibration is noisier in the centroid.
        ``used_fallback`` is true whenever the geometric estimate contributed
        to the returned pose (including a full fallback).
        """
        if not 0.0 <= float(geometry_weight) <= 1.0:
            raise ValueError("geometry_weight must be in [0,1]")
        if disagreement_gate_m <= 0:
            raise ValueError("disagreement_gate_m must be positive")
        learned = self.detect_one(rgb, depth_m)
        learned_by_label = {item.label: item for item in learned}
        geometric = (fallback or CompactColorDepthDetector()).detect_scene(
            rgb, depth_m, calibration=calibration
        )
        geometric_by_label = {item.label: item for item in geometric}
        if set(learned_by_label) == set(self.labels) and set(geometric_by_label) == set(self.labels):
            fused = []
            used = False
            weight = float(geometry_weight)
            for label in self.labels:
                learned_item = learned_by_label[label]
                geometry_item = geometric_by_label[label]
                learned_xy = np.asarray(learned_item.position_xyz_m[:2], dtype=np.float64)
                geometry_xy = np.asarray(geometry_item.position_xyz_m[:2], dtype=np.float64)
                disagreement = float(np.linalg.norm(learned_xy - geometry_xy))
                used |= weight > 0.0 and (disagreement > disagreement_gate_m or weight < 1.0)
                xy = (1.0 - weight) * learned_xy + weight * geometry_xy
                fused.append(Detection(
                    label=label,
                    position_xyz_m=(float(xy[0]), float(xy[1]), float(learned_item.position_xyz_m[2])),
                    yaw_rad=learned_item.yaw_rad,
                    confidence=min(learned_item.confidence, geometry_item.confidence),
                ))
            return tuple(fused), bool(used)
        if set(geometric_by_label) == set(self.labels):
            return tuple(geometric_by_label[label] for label in self.labels), True
        return learned, False

    def scene_state(
        self,
        rgb,
        depth_m,
        *,
        held_label: str | None = None,
        adapter: VisionStateAdapter | None = None,
    ) -> SceneState:
        """Convert one frame directly into the stable :class:`SceneState` contract."""
        resolver = adapter if adapter is not None else VisionStateAdapter()
        return resolver.from_detections(
            self.detect_one(rgb, depth_m), held_label=held_label
        )
