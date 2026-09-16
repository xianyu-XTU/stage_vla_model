"""Explicit registry for replaceable vision providers."""

from collections.abc import Mapping

from stage_vla_v7.interfaces import VisionProvider
from stage_vla_v7.plugins import ProviderRegistry


class VisionProviderRegistry(ProviderRegistry[VisionProvider]):
    def __init__(self) -> None:
        super().__init__("vision")

    def create(
        self,
        name: str,
        config: Mapping[str, object] | None = None,
    ) -> VisionProvider:
        provider = super().create(name, config)
        if not isinstance(provider, VisionProvider):
            raise TypeError("vision factory must return VisionProvider")
        return provider
