"""Deterministic provider composed from independent language parsers."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping

from stage_vla_v7.interfaces import (
    LanguageRequest,
    LanguageResult,
    ModelDescriptor,
    StackRelation,
    TaskPlan,
    UnsupportedTaskError,
)

from ..parser import (
    parse_chinese_clause,
    parse_dsl_clause,
    parse_english_clause,
    phrase_key,
    split_clauses,
)


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


class DeterministicLanguageProvider:
    descriptor = ModelDescriptor(
        name="deterministic-instruction-parser",
        version="1",
        kind="language",
        capabilities=("chinese", "english", "stack-dsl", "multi-relation"),
    )

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        merged = {phrase_key(alias): label for alias, label in DEFAULT_ALIASES.items()}
        for alias, label in (aliases or {}).items():
            if not isinstance(alias, str) or not isinstance(label, str) or not label.strip():
                raise ValueError("language aliases must map strings to non-empty labels")
            merged[phrase_key(alias)] = label.strip()
        self.aliases = merged

    def _resolve(self, phrase: str, available: tuple[str, ...]) -> str:
        cleaned = unicodedata.normalize("NFKC", phrase).strip()
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:上面|顶部|顶上|之上|上|上方)$", "", cleaned).strip()
        normalized = phrase_key(cleaned)
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
        pair = (
            parse_dsl_clause(clause)
            or parse_english_clause(clause)
            or parse_chinese_clause(clause)
        )
        if pair is None:
            raise UnsupportedTaskError(
                "expected STACK(...), object>target, 'stack A on B', or '把A放到B上'"
            )
        return StackRelation(self._resolve(pair[0], available), self._resolve(pair[1], available))

    def interpret(self, request: LanguageRequest) -> LanguageResult:
        clauses = split_clauses(request.text)
        if not clauses:
            raise UnsupportedTaskError("instruction contains no relation")
        relations = tuple(
            self._relation(clause, request.available_object_labels) for clause in clauses
        )
        return LanguageResult(
            TaskPlan(relations, source_text=request.text),
            self.descriptor,
            {"relation_count": len(relations), "learned_model": False},
        )
