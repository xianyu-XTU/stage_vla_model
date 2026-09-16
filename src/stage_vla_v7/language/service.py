"""Validated entry point for language interpretation."""

from __future__ import annotations

from stage_vla_v7.interfaces import (
    LanguageProvider,
    LanguageRequest,
    LanguageResult,
    ProviderError,
)


class LanguageService:
    """Own a selected language provider behind one stable API."""

    def __init__(self, provider: LanguageProvider) -> None:
        if not isinstance(provider, LanguageProvider):
            raise TypeError("provider must implement LanguageProvider")
        self.provider = provider

    def interpret(self, request: LanguageRequest | str) -> LanguageResult:
        normalized = LanguageRequest(request) if isinstance(request, str) else request
        result = self.provider.interpret(normalized)
        if not isinstance(result, LanguageResult):
            raise ProviderError("language provider returned a non-LanguageResult value")
        if result.provider != self.provider.descriptor:
            raise ProviderError("language result descriptor does not match configured provider")
        return result
