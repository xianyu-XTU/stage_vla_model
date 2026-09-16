"""Shared text normalization for deterministic parsers."""

from __future__ import annotations

import re
import unicodedata


def phrase_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[\s_-]+", "_", normalized)
    return normalized.strip("_.,!?;:，。！？；：")


def split_clauses(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if normalized.upper().startswith("STACK(") and normalized.endswith(")"):
        return (normalized,)
    if normalized.upper().startswith("STACK_CHAIN(") and normalized.endswith(")"):
        normalized = normalized[len("STACK_CHAIN(") : -1]
    normalized = re.sub(r"[,，;；]+", "|", normalized)
    normalized = re.sub(r"(?:然后|接着|随后|之后|并且|再)\s*", "|", normalized)
    return tuple(part.strip() for part in normalized.split("|") if part.strip())
