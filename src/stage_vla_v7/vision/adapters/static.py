"""Compatibility imports for providers moved to :mod:`vision.providers`."""

from ..providers.static_provider import CallableVisionProvider, StaticVisionProvider

__all__ = ["CallableVisionProvider", "StaticVisionProvider"]
