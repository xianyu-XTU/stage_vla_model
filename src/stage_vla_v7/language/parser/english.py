"""Constrained English stack-command parsing."""

from __future__ import annotations

import re


_ENGLISH_RE = re.compile(
    r"^(?:please\s+)?(?:(?:first|then|next)\s+)?"
    r"(?:stack|place|put|move|set|position)\s+(?:the\s+)?(?P<object>.+?)\s+"
    r"(?:on\s+top\s+of|on|onto|over|above)\s+(?:the\s+)?(?P<target>.+?)$",
    re.IGNORECASE,
)


def parse_english_clause(clause: str) -> tuple[str, str] | None:
    match = _ENGLISH_RE.fullmatch(clause)
    if match is None:
        return None
    return match.group("object"), match.group("target")
