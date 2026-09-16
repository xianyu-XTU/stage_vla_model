"""Immutable result of binding language semantics to a visual scene."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.action import TaskScheduler
from stage_vla_v7.interfaces import LanguageResult, SkillToken, VisionResult


@dataclass(frozen=True)
class PreparedTask:
    language: LanguageResult
    vision: VisionResult
    tokens: tuple[SkillToken, ...]

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        if tokens != TaskScheduler().schedule(self.language.plan):
            raise ValueError("prepared task tokens do not match the registered skill schedule")
        object.__setattr__(self, "tokens", tokens)
