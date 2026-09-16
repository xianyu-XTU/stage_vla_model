"""Adapter for semantic models that return validated task plans."""

from __future__ import annotations

from collections.abc import Callable

from stage_vla_v7.interfaces import (
    LanguageRequest,
    LanguageResult,
    ModelDescriptor,
    TaskPlan,
)


class CallableLanguageProvider:
    def __init__(
        self,
        function: Callable[[LanguageRequest], TaskPlan],
        *,
        name: str,
        version: str,
        capabilities: tuple[str, ...] = ("structured-task",),
    ) -> None:
        self.function = function
        self.descriptor = ModelDescriptor(name, version, "language", capabilities)

    def interpret(self, request: LanguageRequest) -> LanguageResult:
        plan = self.function(request)
        if not isinstance(plan, TaskPlan):
            raise TypeError("language callable must return TaskPlan")
        return LanguageResult(plan, self.descriptor, {"learned_model": True})
