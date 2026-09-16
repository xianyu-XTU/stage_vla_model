"""Constrained Chinese stack-command parsing."""

from __future__ import annotations

import re


_CHINESE_RE = re.compile(
    r"^(?:请\s*)?(?:(?:先|然后|接着|再)\s*)?(?:把|将)?\s*(?P<object>.+?)\s*"
    r"(?:叠放到|叠放在|堆放到|堆放在|堆叠到|堆叠在|放置到|放置在|"
    r"移动到|移到|放到|放在|置于|叠在|堆在)\s*(?P<target>.+?)$"
)


def parse_chinese_clause(clause: str) -> tuple[str, str] | None:
    match = _CHINESE_RE.fullmatch(clause)
    if match is None:
        return None
    return match.group("object"), match.group("target")
