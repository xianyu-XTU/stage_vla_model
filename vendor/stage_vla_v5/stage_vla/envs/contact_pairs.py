"""Pure helpers for raw contact-pair identity matching.

Isaac Sim 6.0.1 raw contact records expose ``body0`` and ``body1`` IDs.
The repaired M3.1 diagnostic creates:
- one raw sensor scoped to finger A,
- one raw sensor scoped to finger B,
- one raw sensor scoped to red cube (cube_2).

If a finger sensor and the red-cube sensor report the same unordered body-ID
pair in the same physics step, both scoped sensors observed the same physical
contact event. This preserves the original M3 semantic goal ("this finger is
touching the red cube") without relying on the failed filtered force matrix.

This file has no Isaac imports.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

ContactPair = tuple[int, int]


def unordered_pair(body0: int, body1: int) -> ContactPair:
    """Return a deterministic unordered pair key."""
    a = int(body0)
    b = int(body1)
    return (a, b) if a <= b else (b, a)


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def raw_contact_pairs(records: Iterable[Any]) -> set[ContactPair]:
    """Convert raw contact records to a de-duplicated unordered pair set."""
    pairs: set[ContactPair] = set()
    for record in records:
        pairs.add(unordered_pair(_field(record, "body0"), _field(record, "body1")))
    return pairs


def shared_contact_pairs(records_a: Iterable[Any], records_b: Iterable[Any]) -> set[ContactPair]:
    """Return body pairs simultaneously reported by both scoped sensors."""
    return raw_contact_pairs(records_a) & raw_contact_pairs(records_b)
