"""Rigid cube simulation model and VLA object-profile adapter."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import ObjectProfile, SimulationModelDescriptor

from ...physics import PhysicsMaterial, RigidBodyLimits


@dataclass(frozen=True)
class CubeModel:
    size_m: float = 0.04
    mass_kg: float = 0.05
    color_rgb: tuple[int, int, int] = (220, 40, 40)
    material: PhysicsMaterial = PhysicsMaterial()
    rigid_body_limits: RigidBodyLimits = RigidBodyLimits()
    collision_enabled: bool = True
    descriptor: SimulationModelDescriptor = SimulationModelDescriptor(
        "rigid-cube-4cm-50g",
        "object",
        "1",
        "isaac-lab",
        ("rigid", "stackable", "parallel-jaw"),
    )

    def __post_init__(self) -> None:
        if self.size_m <= 0.0 or self.mass_kg <= 0.0:
            raise ValueError("cube size and mass must be positive")
        if len(self.color_rgb) != 3 or any(not 0 <= value <= 255 for value in self.color_rgb):
            raise ValueError("cube RGB color must contain three bytes")

    def object_profile(self, label: str) -> ObjectProfile:
        return ObjectProfile(
            label=label,
            geometry="box",
            size_xyz_m=(self.size_m,) * 3,
            mass_kg=self.mass_kg,
            friction_coefficient=self.material.dynamic_friction,
            stackable=True,
        )
