"""Concrete providers behind the public vision interface."""

from .legacy_provider import LegacyDetectorAdapter
from .static_provider import CallableVisionProvider, StaticVisionProvider

__all__ = ["CallableVisionProvider", "LegacyDetectorAdapter", "StaticVisionProvider"]
