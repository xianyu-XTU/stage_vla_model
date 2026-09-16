"""Vision-independent scene contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..errors import ContractError
from ._validation import finite_tuple


@dataclass(frozen=True)
class ObjectDetection:
    """One object pose estimated in the robot-root coordinate frame."""

    label: str
    position_xyz_m: tuple[float, float, float]
    yaw_rad: float = 0.0
    confidence: float = 1.0
    size_xyz_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ContractError("detection label must be non-empty")
        object.__setattr__(
            self,
            "position_xyz_m",
            finite_tuple(self.position_xyz_m, 3, "position_xyz_m"),
        )
        if not math.isfinite(float(self.yaw_rad)):
            raise ContractError("yaw_rad must be finite")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ContractError("confidence must be in [0, 1]")
        if self.size_xyz_m is not None:
            size = finite_tuple(self.size_xyz_m, 3, "size_xyz_m")
            if any(value <= 0.0 for value in size):
                raise ContractError("size_xyz_m values must be positive")
            object.__setattr__(self, "size_xyz_m", size)


@dataclass(frozen=True)
class SceneState:
    """Validated detector output independent of any particular task."""

    detections: tuple[ObjectDetection, ...]
    frame_id: str | None = None
    timestamp_s: float | None = None
    held_label: str | None = None

    def __post_init__(self) -> None:
        detections = tuple(self.detections)
        if not detections:
            raise ContractError("scene must contain at least one detection")
        labels = [item.label for item in detections]
        if len(labels) != len(set(labels)):
            raise ContractError("scene contains duplicate detection labels")
        if self.timestamp_s is not None and not math.isfinite(float(self.timestamp_s)):
            raise ContractError("timestamp_s must be finite")
        if self.held_label is not None and self.held_label not in labels:
            raise ContractError("held_label must name a detected object")
        object.__setattr__(self, "detections", detections)

    def detection(self, label: str) -> ObjectDetection:
        for item in self.detections:
            if item.label == label:
                return item
        raise ContractError(f"scene is missing required detection {label!r}")
