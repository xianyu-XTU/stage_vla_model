from __future__ import annotations

import pytest

from stage_vla_v7.plugins import ProviderRegistry


def test_provider_registry_is_explicit_and_replace_safe() -> None:
    registry = ProviderRegistry[str]("vision")
    registry.register("demo", lambda config: str(config["value"]))
    assert registry.create("demo", {"value": 3}) == "3"
    with pytest.raises(ValueError, match="already registered"):
        registry.register("demo", lambda _config: "other")
