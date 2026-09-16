"""Adapter for existing V5 compact or YOLO RGB-D detectors."""

from __future__ import annotations

from typing import Mapping

from stage_vla_v7.interfaces import (
    ModelDescriptor,
    ObjectDetection,
    ProviderError,
    SceneState,
    VisionRequest,
    VisionResult,
)


class LegacyDetectorAdapter:
    """Wrap a detector exposing ``detect_one`` or a compatible method."""

    def __init__(
        self,
        detector: object,
        *,
        name: str,
        version: str,
        method: str = "detect_one",
        method_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        function = getattr(detector, method, None)
        if not callable(function):
            raise TypeError(f"legacy detector does not provide callable {method!r}")
        self.detector = detector
        self.method = method
        self.method_kwargs = dict(method_kwargs or {})
        self.descriptor = ModelDescriptor(
            name=name,
            version=version,
            kind="vision",
            capabilities=("legacy-adapter", "rgbd"),
        )

    @staticmethod
    def _convert(item: object) -> ObjectDetection:
        if isinstance(item, ObjectDetection):
            return item
        try:
            return ObjectDetection(
                label=str(getattr(item, "label")),
                position_xyz_m=tuple(getattr(item, "position_xyz_m")),
                yaw_rad=float(getattr(item, "yaw_rad", 0.0)),
                confidence=float(getattr(item, "confidence", 1.0)),
                size_xyz_m=getattr(item, "size_xyz_m", None),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError("legacy detector emitted an invalid detection") from exc

    def detect(self, request: VisionRequest) -> VisionResult:
        function = getattr(self.detector, self.method)
        kwargs = dict(self.method_kwargs)
        kwargs.update(request.calibration)
        raw = function(request.rgb, request.depth_m, **kwargs)
        used_fallback = False
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], bool):
            raw, used_fallback = raw
        try:
            detections = tuple(self._convert(item) for item in raw)
        except TypeError as exc:
            raise ProviderError("legacy detector output is not iterable") from exc
        scene = SceneState(
            detections,
            frame_id=request.frame_id,
            timestamp_s=request.timestamp_s,
            held_label=request.metadata.get("held_label"),
        )
        return VisionResult(
            scene,
            self.descriptor,
            {"used_fallback": used_fallback, "legacy_method": self.method},
        )
