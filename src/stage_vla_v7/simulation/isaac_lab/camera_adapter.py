"""Translate raw Isaac camera payloads into the public vision request."""

from __future__ import annotations

from collections.abc import Mapping

from stage_vla_v7.interfaces import VisionRequest


class IsaacCameraAdapter:
    def to_vision_request(
        self,
        rgb: object,
        depth_m: object | None = None,
        *,
        frame_id: str | None = None,
        timestamp_s: float | None = None,
        calibration: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> VisionRequest:
        details = {"backend": "isaac-lab", **dict(metadata or {})}
        return VisionRequest(
            rgb=rgb,
            depth_m=depth_m,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            calibration=dict(calibration or {}),
            metadata=details,
        )
