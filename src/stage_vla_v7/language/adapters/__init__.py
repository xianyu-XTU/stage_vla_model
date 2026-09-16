"""Built-in language adapters."""

from .callable import CallableLanguageProvider
from .deterministic import DeterministicLanguageProvider

__all__ = ["CallableLanguageProvider", "DeterministicLanguageProvider"]
