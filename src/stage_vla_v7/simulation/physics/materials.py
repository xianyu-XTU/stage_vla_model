"""Simulator-neutral rigid-body material descriptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsMaterial:
    static_friction: float = 0.8
    dynamic_friction: float = 0.8
    restitution: float = 0.0

    def __post_init__(self) -> None:
        if min(self.static_friction, self.dynamic_friction, self.restitution) < 0.0:
            raise ValueError("material coefficients must be non-negative")
        if self.restitution > 1.0:
            raise ValueError("restitution must be in [0, 1]")
