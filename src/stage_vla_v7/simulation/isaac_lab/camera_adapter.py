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

    def rgb_u8_batch(
        self,
        camera: object,
        *,
        camera_name: str = VISION_CAMERA_NAME,
    ) -> Any:
        """Read Isaac RGB output as contiguous uint8 ``[N,H,W,3]``."""
        output = self._camera_output(camera)
        if "rgb" not in output:
            raise RuntimeError(f"camera {camera_name!r} has no rgb output")
        torch = self._torch()
        rgb = self._to_torch(output["rgb"])
        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise RuntimeError(f"unexpected batched camera RGB shape: {tuple(rgb.shape)}")
        rgb = rgb[..., :3]
        if torch.is_floating_point(rgb):
            max_value = float(rgb.max().item()) if rgb.numel() else 0.0
            if max_value <= 1.0 + 1e-6:
                rgb = rgb * 255.0
            rgb = rgb.round().clamp(0.0, 255.0).to(torch.uint8)
        else:
            rgb = rgb.to(torch.uint8)
        return self._numpy().ascontiguousarray(rgb.detach().cpu().numpy())

    def depth_m_batch(
        self,
        camera: object,
        *,
        camera_name: str = VISION_CAMERA_NAME,
    ) -> Any:
        """Read Isaac image-plane distance as contiguous float32 ``[N,H,W]``."""
        output = self._camera_output(camera)
        key = "distance_to_image_plane"
        if key not in output:
            raise RuntimeError(f"camera {camera_name!r} has no {key} output")
        depth = self._to_torch(output[key])
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim != 3:
            raise RuntimeError(f"unexpected batched camera depth shape: {tuple(depth.shape)}")
        numpy = self._numpy()
        return numpy.ascontiguousarray(
            depth.detach().cpu().numpy().astype(numpy.float32, copy=False)
        )

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

    @staticmethod
    def _camera_output(camera: object) -> Mapping[str, object]:
        data = getattr(camera, "data", None)
        output = getattr(data, "output", None)
        if not isinstance(output, Mapping):
            raise RuntimeError("Isaac camera does not expose an output mapping")
        return output

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - Isaac Lab supplies torch
            raise RuntimeError("Isaac camera conversion requires torch") from exc
        return torch

    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy
        except ImportError as exc:  # pragma: no cover - Isaac Lab supplies numpy
            raise RuntimeError("Isaac camera conversion requires numpy") from exc
        return numpy

    @classmethod
    def _to_torch(cls, value: object) -> Any:
        if hasattr(value, "torch"):
            value = getattr(value, "torch")
        return cls._torch().as_tensor(value)
