"""Translate raw Isaac camera payloads into the public vision request."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from stage_vla_v7.interfaces import VisionRequest


VISION_CAMERA_NAME = "v7_multicube_camera"
OBSERVER_CAMERA_NAME = "v7_observer_camera"


@dataclass(frozen=True)
class CameraBindings:
    """Distinct camera handles; the observer handle never enters VisionService."""

    vision: object | None
    observer: object | None


class IsaacCameraAdapter:
    def bind(
        self,
        scene: Mapping[str, object] | Any,
        *,
        use_vision: bool,
        record_video: bool,
        vision_name: str = VISION_CAMERA_NAME,
        observer_name: str = OBSERVER_CAMERA_NAME,
    ) -> CameraBindings:
        if vision_name == observer_name:
            raise ValueError("Vision and Observer camera IDs must be distinct")
        vision = scene[vision_name] if use_vision else None
        observer = scene[observer_name] if record_video else None
        if use_vision and vision is None:
            raise LookupError(f"Vision camera {vision_name!r} is unavailable")
        if record_video and observer is None:
            raise LookupError(f"Observer camera {observer_name!r} is unavailable")
        return CameraBindings(vision, observer)

    def observer_rgb_payload(self, camera: object, *, environment_index: int) -> object:
        """Read only the observer RGB payload without importing torch."""
        data = getattr(camera, "data", None)
        output = getattr(data, "output", None)
        if not isinstance(output, Mapping) or "rgb" not in output:
            raise ValueError("Observer camera does not expose an RGB output")
        return output["rgb"][int(environment_index), ..., :3]

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
