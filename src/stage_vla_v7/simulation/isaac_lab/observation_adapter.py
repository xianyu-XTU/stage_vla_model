"""Convert Isaac-style tensor rows into dependency-free observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from stage_vla_v7.interfaces import RobotObservation


def _plain_values(value: Any) -> object:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "tolist"):
        current = current.tolist()
    return current


class IsaacObservationAdapter:
    """Validate simulator observations before they enter the VLA core."""

    def __init__(self, schema: str = "stage-vla-v7.isaac-lab.v5-policy") -> None:
        if not schema.strip():
            raise ValueError("observation schema must be non-empty")
        self.schema = schema

    def to_robot_observation(
        self,
        row: object,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> RobotObservation:
        values = _plain_values(row)
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
            raise ValueError("Isaac observation row must be a rank-1 numeric sequence")
        materialized = tuple(values)
        if any(
            isinstance(item, Iterable) and not isinstance(item, (str, bytes))
            for item in materialized
        ):
            raise ValueError("Isaac observation row must be rank 1")
        return RobotObservation(materialized, self.schema, dict(metadata or {}))

    def batch(self, observations: object) -> tuple[RobotObservation, ...]:
        rows = _plain_values(observations)
        if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, Mapping)):
            raise ValueError("Isaac observation batch must be a rank-2 numeric sequence")
        result = tuple(
            self.to_robot_observation(row, metadata={"batch_index": index})
            for index, row in enumerate(rows)
        )
        if not result:
            raise ValueError("Isaac observation batch must not be empty")
        return result
