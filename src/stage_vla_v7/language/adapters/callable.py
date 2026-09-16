"""Adapter for LLM clients that already return a validated task plan."""

from __future__ import annotations

from collections.abc import Callable

from stage_vla_v7.contracts import ModelDescriptor, TaskPlan

from ..interfaces import LanguageRequest, LanguageResult


class CallableLanguageProvider:
    """Wrap a local or remote semantic model without coupling to its SDK."""

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
