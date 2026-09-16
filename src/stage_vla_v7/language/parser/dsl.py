"""Strict stack DSL parsing."""

from __future__ import annotations

import re


_STRICT_RE = re.compile(
    r"^STACK\s*\(\s*object\s*=\s*(?P<object>[A-Za-z0-9_-]+)\s*,\s*"
    r"target\s*=\s*(?P<target>[A-Za-z0-9_-]+)\s*\)$",
    re.IGNORECASE,
)
_BARE_RE = re.compile(r"^(?P<object>[A-Za-z0-9_-]+)\s*>\s*(?P<target>[A-Za-z0-9_-]+)$")


def parse_dsl_clause(clause: str) -> tuple[str, str] | None:
    match = _STRICT_RE.fullmatch(clause) or _BARE_RE.fullmatch(clause)
    if match is None:
        return None
    return match.group("object"), match.group("target")
