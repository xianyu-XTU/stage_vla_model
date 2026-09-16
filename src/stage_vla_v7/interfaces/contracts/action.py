"""Normalized robot action contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

from ..errors import ContractError
from ._validation import finite_tuple


@dataclass(frozen=True)
class RobotAction:
    """Normalized command in the frozen [dx, dy, dz, dyaw, grip] order."""

    dx: float
    dy: float
    dz: float
    dyaw: float
    grip: float

    def __post_init__(self) -> None:
        finite_tuple(self.values, 5, "robot action")

    @property
    def values(self) -> tuple[float, float, float, float, float]:
        return (self.dx, self.dy, self.dz, self.dyaw, self.grip)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "RobotAction":
        return cls(*finite_tuple(values, 5, "robot action"))

    def clipped(self, low: float = -1.0, high: float = 1.0) -> "RobotAction":
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ContractError("invalid action clipping bounds")
        return RobotAction.from_values(min(high, max(low, value)) for value in self.values)
