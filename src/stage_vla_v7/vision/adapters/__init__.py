"""Built-in vision adapters."""

from .legacy import LegacyDetectorAdapter
from .static import CallableVisionProvider, StaticVisionProvider

__all__ = ["CallableVisionProvider", "LegacyDetectorAdapter", "StaticVisionProvider"]
