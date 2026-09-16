"""Validated object-state interface between a compact vision model and policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ..objects import ObjectModule


@dataclass(frozen=True)
class Detection:
    """One compact-model detection in the robot root frame."""

    label: str
    position_xyz_m: tuple[float, float, float]
    yaw_rad: float
    confidence: float

    def validate(self) -> None:
        if not self.label or self.label.strip() != self.label:
            raise ValueError("label must be a non-empty trimmed string")
        if len(self.position_xyz_m) != 3:
            raise ValueError("position_xyz_m must have length 3")
        values = (*self.position_xyz_m, self.yaw_rad, self.confidence)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("detection contains NaN or Inf")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class SceneState:
    """The minimum object state required by the v5 skill adapters."""

    object_detection: Detection
    target_detection: Detection
    held_label: str | None

    def validate(self) -> None:
        self.object_detection.validate()
        self.target_detection.validate()
        if self.object_detection.label == self.target_detection.label:
            raise ValueError("object and target detections must have different labels")
        if self.held_label is not None and not self.held_label:
            raise ValueError("held_label must be non-empty when provided")


@dataclass(frozen=True)
class VisionStateAdapter:
    """Resolve detector outputs into the named object/target state contract."""

    object_label: str = "red_cube"
    target_label: str = "blue_cube"

    @classmethod
    def for_module(cls, module: ObjectModule) -> "VisionStateAdapter":
        if not isinstance(module, ObjectModule):
            raise TypeError("module must be an ObjectModule")
        return cls(module.object_label, module.support_label)

    def from_detections(self, detections: Iterable[Detection], *, held_label: str | None = None) -> SceneState:
        by_label: dict[str, Detection] = {}
        for detection in detections:
            detection.validate()
            if detection.label in by_label:
                raise ValueError(f"duplicate detection label: {detection.label}")
            by_label[detection.label] = detection
        try:
            state = SceneState(
                object_detection=by_label[self.object_label],
                target_detection=by_label[self.target_label],
                held_label=held_label,
            )
        except KeyError as exc:
            raise ValueError(f"missing required detection: {exc.args[0]}") from exc
        state.validate()
        return state
