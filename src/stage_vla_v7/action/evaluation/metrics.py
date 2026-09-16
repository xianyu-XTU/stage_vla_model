"""Dependency-free evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return center - radius, center + radius


@dataclass(frozen=True)
class SuccessMetrics:
    successes: int
    trials: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def confidence_95(self) -> tuple[float, float] | None:
        return wilson_interval(self.successes, self.trials) if self.trials else None
