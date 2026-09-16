"""Small explicit registry for provider factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Mapping, TypeVar


ProviderT = TypeVar("ProviderT")
Factory = Callable[[Mapping[str, object]], ProviderT]


class ProviderRegistry(Generic[ProviderT]):
    """Register providers by stable configuration name."""

    def __init__(self, kind: str) -> None:
        if not kind.strip():
            raise ValueError("registry kind must be non-empty")
        self.kind = kind
        self._factories: dict[str, Factory[ProviderT]] = {}

    def register(
        self,
        name: str,
        factory: Factory[ProviderT],
        *,
        replace: bool = False,
    ) -> None:
        if not name.strip():
            raise ValueError("provider name must be non-empty")
        if name in self._factories and not replace:
            raise ValueError(f"{self.kind} provider {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, config: Mapping[str, object] | None = None) -> ProviderT:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            choices = ", ".join(self.names) or "<none>"
            raise LookupError(
                f"unknown {self.kind} provider {name!r}; available: {choices}"
            ) from exc
        return factory(dict(config or {}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
