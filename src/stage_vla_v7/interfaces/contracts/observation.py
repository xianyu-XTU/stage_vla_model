"""Dependency-free robot observation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..errors import ContractError
from ._validation import finite_tuple


@dataclass(frozen=True)
class RobotObservation:
    """A finite numeric observation with an explicit schema identity."""

    values: tuple[float, ...]
    schema: str = "stage-vla-v7.generic"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ContractError("robot observation schema must be non-empty")
        object.__setattr__(self, "values", finite_tuple(self.values, None, "robot observation"))
