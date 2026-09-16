"""Physical object profile contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..errors import ContractError
from ._validation import finite_tuple


@dataclass(frozen=True)
class ObjectProfile:
    """Physical metadata resolved outside vision and language providers."""

    label: str
    geometry: str
    size_xyz_m: tuple[float, float, float]
    mass_kg: float
    grasp_mode: str = "parallel_jaw"
    support_mode: str = "flat"
    friction_coefficient: float = 0.8
    deformable: bool = False
    stackable: bool = True

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.geometry.strip():
            raise ContractError("object label and geometry must be non-empty")
        size = finite_tuple(self.size_xyz_m, 3, "size_xyz_m")
        if any(value <= 0.0 for value in size):
            raise ContractError("object size values must be positive")
        object.__setattr__(self, "size_xyz_m", size)
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ContractError("mass_kg must be positive")
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient <= 0.0:
            raise ContractError("friction_coefficient must be positive")
