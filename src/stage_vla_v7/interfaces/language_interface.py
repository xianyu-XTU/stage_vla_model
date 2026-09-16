"""Public language boundary with no model or simulator dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from .contracts import ModelDescriptor, TaskPlan


@dataclass(frozen=True)
class LanguageRequest:
    text: str
    available_object_labels: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("language request text must be non-empty")
        object.__setattr__(self, "available_object_labels", tuple(self.available_object_labels))


@dataclass(frozen=True)
class LanguageResult:
    plan: TaskPlan
    provider: ModelDescriptor
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class LanguageProvider(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...

    def interpret(self, request: LanguageRequest) -> LanguageResult: ...
