"""Concrete language providers."""

from .callable import CallableLanguageProvider
from .deterministic import DEFAULT_ALIASES, DeterministicLanguageProvider

__all__ = ["CallableLanguageProvider", "DEFAULT_ALIASES", "DeterministicLanguageProvider"]
