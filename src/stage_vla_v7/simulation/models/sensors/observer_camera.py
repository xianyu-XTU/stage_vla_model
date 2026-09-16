"""Third-person RGB camera used only for human-observer recordings."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from stage_vla_v7.interfaces import SimulationModelDescriptor

from ...config import CameraSpec


@dataclass(frozen=True)
class ObserverCameraModel:
    name: str = "v7_observer_camera"
    width: int = 640
    height: int = 480
    fps: float = 20.0
    position_m: tuple[float, float, float] = (1.0, 0.0, 0.40)
    rotation_wxyz: tuple[float, float, float, float] = (
        -0.61237,
        -0.61237,
        0.35355,
        0.35355,
    )
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 20.955
    descriptor: SimulationModelDescriptor = SimulationModelDescriptor(
        "stack-observer-camera",
        "sensor",
        "1",
        "isaac-lab",
        ("rgb", "observer-only", "no-control-input"),
    )

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.fps)) or self.fps <= 0.0:
            raise ValueError("observer camera fps must be finite and positive")
        self.to_camera_spec()

    def to_camera_spec(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> CameraSpec:
        capture_fps = self.fps if fps is None else float(fps)
        if not math.isfinite(capture_fps) or capture_fps <= 0.0:
            raise ValueError("observer camera fps must be finite and positive")
        return CameraSpec(
            name=self.name,
            role="observer",
            width=self.width if width is None else int(width),
            height=self.height if height is None else int(height),
            data_types=("rgb",),
            position_m=self.position_m,
            rotation_wxyz=self.rotation_wxyz,
            focal_length_mm=self.focal_length_mm,
            horizontal_aperture_mm=self.horizontal_aperture_mm,
            update_period_s=1.0 / capture_fps,
        )

    @classmethod
    def from_json(cls, path: Path) -> "ObserverCameraModel":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != "stage_vla_v7.observer_camera.v1":
            raise ValueError("unsupported observer camera configuration schema")
        return cls(
            name=str(payload.get("name", "v7_observer_camera")),
            width=int(payload["width"]),
            height=int(payload["height"]),
            fps=float(payload["fps"]),
            position_m=tuple(float(value) for value in payload["position_m"]),
            rotation_wxyz=tuple(
                float(value) for value in payload["rotation_wxyz"]
            ),
            focal_length_mm=float(payload["focal_length_mm"]),
            horizontal_aperture_mm=float(payload["horizontal_aperture_mm"]),
        )
