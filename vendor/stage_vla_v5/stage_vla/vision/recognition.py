"""Replaceable object-recognition boundary for current and future vision models.

Recognition backends consume an opaque :class:`VisionFrame` and emit only
structured detections.  Task compilation and continuous action generation are
deliberately outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Protocol, Sequence

from ..objects import ObjectModule, get_object_spec
from .compact_model import LearnedRGBDDetector
from .rgbd_detector import CameraCalibration, CompactColorDepthDetector
from .state import Detection, SceneState, VisionStateAdapter


@dataclass(frozen=True)
class VisionFrame:
    """Sensor payload passed unchanged to a recognition backend.

    ``rgb`` is the only required channel. Depth and calibration are optional so
    an RGB-only or multimodal model can implement the same backend protocol.
    """

    rgb: object
    depth_m: object | None = None
    calibration: object | None = None
    frame_id: str | None = None
    timestamp_s: float | None = None

    def validate(self) -> None:
        if self.rgb is None:
            raise ValueError("rgb frame is required")
        if self.frame_id is not None and (not self.frame_id or self.frame_id.strip() != self.frame_id):
            raise ValueError("frame_id must be a non-empty trimmed string")
        if self.timestamp_s is not None and not isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")


@dataclass(frozen=True)
class RecognitionRequest:
    """One scene-recognition request with optional task label hints."""

    frame: VisionFrame
    candidate_labels: tuple[str, ...] = ()

    def validate(self) -> None:
        self.frame.validate()
        if len(set(self.candidate_labels)) != len(self.candidate_labels):
            raise ValueError("candidate_labels must not contain duplicates")
        for label in self.candidate_labels:
            get_object_spec(label)


class RecognitionBackend(Protocol):
    """Interface to implement when adding a new visual recognition model."""

    @property
    def name(self) -> str:
        """Stable backend identifier included in diagnostics."""

    def detect(self, request: RecognitionRequest) -> Iterable[Detection]:
        """Return structured detections; never return task tokens or actions."""


@dataclass(frozen=True)
class RecognitionResult:
    """Validated scene-level output independent of the recognition model."""

    detections: tuple[Detection, ...]
    source: str
    frame_id: str | None = None

    def by_label(self) -> dict[str, Detection]:
        return {detection.label: detection for detection in self.detections}

    def require(self, labels: Sequence[str]) -> tuple[Detection, ...]:
        indexed = self.by_label()
        missing = [label for label in labels if label not in indexed]
        if missing:
            raise ValueError(f"missing required detections: {missing}")
        return tuple(indexed[label] for label in labels)

    def scene_state(
        self,
        module: ObjectModule,
        *,
        held_label: str | None = None,
    ) -> SceneState:
        """Resolve the full-scene result into action-policy object/support roles."""
        return VisionStateAdapter.for_module(module).from_detections(
            self.detections,
            held_label=held_label,
        )


class ObjectRecognitionModule:
    """Validate and normalize detections from a replaceable visual backend."""

    def __init__(
        self,
        backend: RecognitionBackend,
        *,
        confidence_floor: float = 0.0,
        require_catalog_labels: bool = True,
    ) -> None:
        if not 0.0 <= float(confidence_floor) <= 1.0:
            raise ValueError("confidence_floor must be in [0,1]")
        self.backend = backend
        self.confidence_floor = float(confidence_floor)
        self.require_catalog_labels = bool(require_catalog_labels)

    def recognize(
        self,
        frame: VisionFrame,
        *,
        candidate_labels: Sequence[str] = (),
        require_all: bool = False,
    ) -> RecognitionResult:
        labels = tuple(candidate_labels)
        request = RecognitionRequest(frame=frame, candidate_labels=labels)
        request.validate()
        source = self.backend.name
        if not source or source.strip() != source:
            raise ValueError("recognition backend name must be a non-empty trimmed string")

        accepted: dict[str, Detection] = {}
        for detection in self.backend.detect(request):
            if not isinstance(detection, Detection):
                raise TypeError("recognition backend must emit Detection instances")
            detection.validate()
            if self.require_catalog_labels:
                get_object_spec(detection.label)
            if detection.confidence < self.confidence_floor:
                continue
            if detection.label in accepted:
                raise ValueError(f"duplicate detection label: {detection.label}")
            accepted[detection.label] = detection

        result = RecognitionResult(
            detections=tuple(accepted.values()),
            source=source,
            frame_id=frame.frame_id,
        )
        if require_all:
            result.require(labels)
        return result


class StaticRecognitionBackend:
    """Dependency-free backend for simulator truth and integration tests."""

    name = "static"

    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = tuple(detections)

    def detect(self, request: RecognitionRequest) -> Iterable[Detection]:
        del request
        return self._detections


class CompactRGBDRecognitionBackend:
    """Adapter exposing the existing calibrated detector through the protocol."""

    name = "compact_color_depth"

    def __init__(self, detector: CompactColorDepthDetector | None = None) -> None:
        self.detector = detector or CompactColorDepthDetector()

    def detect(self, request: RecognitionRequest) -> Iterable[Detection]:
        frame = request.frame
        if frame.depth_m is None:
            raise ValueError("compact RGB-D recognition requires depth_m")
        if not isinstance(frame.calibration, CameraCalibration):
            raise ValueError("compact RGB-D recognition requires CameraCalibration")
        labels = request.candidate_labels or ("red_cube", "blue_cube")
        return self.detector.detect_scene(
            frame.rgb,
            frame.depth_m,
            calibration=frame.calibration,
            labels=labels,
        )


class LearnedRGBDRecognitionBackend:
    """Adapter for the existing learned RGB-D detector implementation."""

    name = "learned_rgbd"

    def __init__(self, detector: LearnedRGBDDetector) -> None:
        if not isinstance(detector, LearnedRGBDDetector):
            raise TypeError("detector must be a LearnedRGBDDetector")
        self.detector = detector

    def detect(self, request: RecognitionRequest) -> Iterable[Detection]:
        frame = request.frame
        if frame.depth_m is None:
            raise ValueError("learned RGB-D recognition requires depth_m")
        unsupported = set(request.candidate_labels) - set(self.detector.labels)
        if unsupported:
            raise ValueError(
                "learned RGB-D model does not support labels: "
                f"{sorted(unsupported)}"
            )
        detections = self.detector.detect_one(frame.rgb, frame.depth_m)
        if not request.candidate_labels:
            return detections
        requested = set(request.candidate_labels)
        return tuple(item for item in detections if item.label in requested)
