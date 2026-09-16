"""Internal validation helpers for dependency-free contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..errors import ContractError


def finite_tuple(values: Iterable[float], size: int | None, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if size is not None and len(result) != size:
        raise ContractError(f"{name} must contain {size} finite values")
    if not result or not all(math.isfinite(value) for value in result):
        expected = size if size is not None else "one or more"
        raise ContractError(f"{name} must contain {expected} finite values")
    return result
