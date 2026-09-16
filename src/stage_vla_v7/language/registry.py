"""Explicit registry for replaceable language providers."""

from collections.abc import Mapping

from stage_vla_v7.interfaces import LanguageProvider
from stage_vla_v7.plugins import ProviderRegistry


class LanguageProviderRegistry(ProviderRegistry[LanguageProvider]):
    def __init__(self) -> None:
        super().__init__("language")

    def create(
        self,
        name: str,
        config: Mapping[str, object] | None = None,
    ) -> LanguageProvider:
        provider = super().create(name, config)
        if not isinstance(provider, LanguageProvider):
            raise TypeError("language factory must return LanguageProvider")
        return provider
