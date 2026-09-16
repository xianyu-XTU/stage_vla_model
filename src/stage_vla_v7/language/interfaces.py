"""Language port definitions. This module has no vision or action dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from stage_vla_v7.contracts import ModelDescriptor, TaskPlan


@dataclass(frozen=True)
class LanguageRequest:
    """Text and optional semantic context accepted by a language provider."""

    text: str
    available_object_labels: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("language request text must be non-empty")


@dataclass(frozen=True)
class LanguageResult:
    """Structured task semantics with provider audit information."""

    plan: TaskPlan
    provider: ModelDescriptor
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class LanguageProvider(Protocol):
    """Replaceable text-to-task boundary."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    def interpret(self, request: LanguageRequest) -> LanguageResult:
        """Return semantic relations and never a continuous robot action."""
