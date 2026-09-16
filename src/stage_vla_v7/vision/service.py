"""Validated entry point for all visual inference."""

from __future__ import annotations

from stage_vla_v7.interfaces import ProviderError, VisionProvider, VisionRequest, VisionResult


class VisionService:
    """Own a selected provider without exposing its implementation downstream."""

    def __init__(self, provider: VisionProvider) -> None:
        if not isinstance(provider, VisionProvider):
            raise TypeError("provider must implement VisionProvider")
        self.provider = provider

    def observe(self, request: VisionRequest) -> VisionResult:
        result = self.provider.detect(request)
        if not isinstance(result, VisionResult):
            raise ProviderError("vision provider returned a non-VisionResult value")
        if result.provider != self.provider.descriptor:
            raise ProviderError("vision result descriptor does not match configured provider")
        return result
