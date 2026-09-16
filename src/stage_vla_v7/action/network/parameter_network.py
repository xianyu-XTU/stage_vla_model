"""The fixed parameter-prediction boundary after skill scheduling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


ACTION_PARAMETER_ORDER = ("dx", "dy", "dz", "dyaw", "grip")
ACTION_DIM = len(ACTION_PARAMETER_ORDER)


class ParameterNetwork(Protocol):
    observation_dim: int

    def predict_parameters(self, observation: Sequence[float]) -> Sequence[float]: ...


def validate_action_parameters(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != ACTION_DIM or not all(math.isfinite(value) for value in result):
        raise ValueError("parameter network must return five finite action values")
    return result
