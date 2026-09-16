"""RGB camera model description."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import SimulationModelDescriptor


@dataclass(frozen=True)
class RGBCameraModel:
    width: int = 128
    height: int = 128
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 20.955
    descriptor: SimulationModelDescriptor = SimulationModelDescriptor(
        "stack-rgb-camera",
        "sensor",
        "1",
        "isaac-lab",
        ("rgb",),
    )

    def __post_init__(self) -> None:
        if min(self.width, self.height) < 1 or min(
            self.focal_length_mm, self.horizontal_aperture_mm
        ) <= 0.0:
            raise ValueError("RGB camera dimensions and optics must be positive")
