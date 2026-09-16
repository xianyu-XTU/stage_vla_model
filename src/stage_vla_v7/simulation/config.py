"""Dependency-light configuration shared by Simulation backends."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CameraSpec:
    """Backend-neutral camera declaration consumed by an environment factory."""

    name: str
    role: str
    width: int
    height: int
    data_types: tuple[str, ...]
    position_m: tuple[float, float, float] = (1.0, 0.0, 0.40)
    rotation_wxyz: tuple[float, float, float, float] = (
        -0.61237,
        -0.61237,
        0.35355,
        0.35355,
    )
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 20.955
    update_period_s: float = 0.05

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("camera name must be non-empty")
        if self.role not in {"vision", "observer"}:
            raise ValueError("camera role must be vision or observer")
        if min(int(self.width), int(self.height)) < 1:
            raise ValueError("camera dimensions must be positive")
        if not self.data_types or any(
            value not in {"rgb", "distance_to_image_plane"}
            for value in self.data_types
        ):
            raise ValueError("camera data types must contain supported outputs")
        if self.role == "vision" and not {
            "rgb",
            "distance_to_image_plane",
        }.issubset(self.data_types):
            raise ValueError("Vision camera must provide RGB and depth")
        if self.role == "observer" and self.data_types != ("rgb",):
            raise ValueError("Observer camera must provide RGB only")
        numeric = (
            *self.position_m,
            *self.rotation_wxyz,
            self.focal_length_mm,
            self.horizontal_aperture_mm,
            self.update_period_s,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("camera pose and optics must be finite")
        if min(
            float(self.focal_length_mm),
            float(self.horizontal_aperture_mm),
            float(self.update_period_s),
        ) <= 0.0:
            raise ValueError("camera optics and update period must be positive")


def validate_camera_specs(specs: tuple[CameraSpec, ...]) -> None:
    names = tuple(spec.name for spec in specs)
    roles = tuple(spec.role for spec in specs)
    if len(names) != len(set(names)):
        raise ValueError("camera IDs must be unique")
    if len(roles) != len(set(roles)):
        raise ValueError("only one camera may own each camera role")
