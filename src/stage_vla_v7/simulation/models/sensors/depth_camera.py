"""Depth camera model description."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import SimulationModelDescriptor


@dataclass(frozen=True)
class DepthCameraModel:
    width: int = 128
    height: int = 128
    data_type: str = "distance_to_image_plane"
    minimum_depth_m: float = 0.01
    maximum_depth_m: float = 5.0
    descriptor: SimulationModelDescriptor = SimulationModelDescriptor(
        "stack-depth-camera",
        "sensor",
        "1",
        "isaac-lab",
        ("depth", "metric"),
    )

    def __post_init__(self) -> None:
        if min(self.width, self.height) < 1:
            raise ValueError("depth camera dimensions must be positive")
        if self.minimum_depth_m <= 0.0 or self.minimum_depth_m >= self.maximum_depth_m:
            raise ValueError("depth range is invalid")
