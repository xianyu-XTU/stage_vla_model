"""Small deterministic Chinese/English stack-command provider."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping

from stage_vla_v7.contracts import (
    ModelDescriptor,
    StackRelation,
    TaskPlan,
    UnsupportedTaskError,
)

from ..interfaces import LanguageRequest, LanguageResult


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[\s_-]+", "_", normalized)
    return normalized.strip("_.,!?;:，。！？；：")


DEFAULT_ALIASES = {
    "红方块": "red_cube",
    "红色方块": "red_cube",
    "红色的方块": "red_cube",
    "蓝方块": "blue_cube",
    "蓝色方块": "blue_cube",
    "蓝色的方块": "blue_cube",
    "绿方块": "green_cube",
    "绿色方块": "green_cube",
    "黄方块": "yellow_cube",
    "黄色方块": "yellow_cube",
    "白色杯子": "white_cup",
    "蓝色杯子": "blue_cup",
    "red cube": "red_cube",
    "blue cube": "blue_cube",
    "green cube": "green_cube",
    "yellow cube": "yellow_cube",
}


_STRICT_RE = re.compile(
    r"^STACK\s*\(\s*object\s*=\s*(?P<object>[A-Za-z0-9_-]+)\s*,\s*"
    r"target\s*=\s*(?P<target>[A-Za-z0-9_-]+)\s*\)$",
    re.IGNORECASE,
)
_ENGLISH_RE = re.compile(
    r"^(?:please\s+)?(?:(?:first|then|next)\s+)?"
    r"(?:stack|place|put|move|set|position)\s+(?:the\s+)?(?P<object>.+?)\s+"
    r"(?:on\s+top\s+of|on|onto|over|above)\s+(?:the\s+)?(?P<target>.+?)$",
    re.IGNORECASE,
)
_CHINESE_RE = re.compile(
    r"^(?:请\s*)?(?:(?:先|然后|接着|再)\s*)?(?:把|将)?\s*(?P<object>.+?)\s*"
    r"(?:叠放到|叠放在|堆放到|堆放在|堆叠到|堆叠在|放置到|放置在|"
    r"移动到|移到|放到|放在|置于|叠在|堆在)\s*(?P<target>.+?)$"
)
_BARE_RE = re.compile(r"^(?P<object>[A-Za-z0-9_-]+)\s*>\s*(?P<target>[A-Za-z0-9_-]+)$")


class DeterministicLanguageProvider:
    """Parse a constrained grammar without loading a language model."""

    descriptor = ModelDescriptor(
        name="deterministic-instruction-parser",
        version="1",
        kind="language",
        capabilities=("chinese", "english", "stack-dsl", "multi-relation"),
    )

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        merged = {_key(alias): label for alias, label in DEFAULT_ALIASES.items()}
        for alias, label in (aliases or {}).items():
            if not isinstance(alias, str) or not isinstance(label, str) or not label.strip():
                raise ValueError("language aliases must map strings to non-empty labels")
            merged[_key(alias)] = label.strip()
        self.aliases = merged

    @staticmethod
    def _split(text: str) -> tuple[str, ...]:
        normalized = unicodedata.normalize("NFKC", text).strip()
        if normalized.upper().startswith("STACK(") and normalized.endswith(")"):
            return (normalized,)
        if normalized.upper().startswith("STACK_CHAIN(") and normalized.endswith(")"):
            normalized = normalized[len("STACK_CHAIN(") : -1]
        normalized = re.sub(r"[,，;；]+", "|", normalized)
        normalized = re.sub(r"(?:然后|接着|随后|之后|并且|再)\s*", "|", normalized)
        return tuple(part.strip() for part in normalized.split("|") if part.strip())

    def _resolve(self, phrase: str, available: tuple[str, ...]) -> str:
        cleaned = unicodedata.normalize("NFKC", phrase).strip()
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:上面|顶部|顶上|之上|上|上方)$", "", cleaned).strip()
        normalized = _key(cleaned)
        resolved = self.aliases.get(normalized, normalized)
        ordinal = re.fullmatch(r"第?(\d+)个?(?:方块|立方体)", cleaned)
        if ordinal:
            resolved = f"cube_{int(ordinal.group(1))}"
        if available and resolved not in available:
            raise UnsupportedTaskError(f"object {resolved!r} is not available in the scene")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", resolved):
            raise UnsupportedTaskError(f"unknown object phrase {phrase!r}")
        return resolved

    def _relation(self, clause: str, available: tuple[str, ...]) -> StackRelation:
        match = _STRICT_RE.fullmatch(clause) or _BARE_RE.fullmatch(clause)
        if match is None:
            match = _ENGLISH_RE.fullmatch(clause) or _CHINESE_RE.fullmatch(clause)
        if match is None:
            raise UnsupportedTaskError(
                "expected STACK(...), object>target, 'stack A on B', or '把A放到B上'"
            )
        return StackRelation(
            self._resolve(match.group("object"), available),
            self._resolve(match.group("target"), available),
        )

    def interpret(self, request: LanguageRequest) -> LanguageResult:
        clauses = self._split(request.text)
        if not clauses:
            raise UnsupportedTaskError("instruction contains no relation")
        relations = tuple(
            self._relation(clause, request.available_object_labels) for clause in clauses
        )
        plan = TaskPlan(relations, source_text=request.text)
        return LanguageResult(
            plan,
            self.descriptor,
            {"relation_count": len(relations), "learned_model": False},
        )
