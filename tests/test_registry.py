from __future__ import annotations

import pytest

from stage_vla_v7.plugins import ProviderRegistry
from stage_vla_v7.contracts import ObjectDetection, SceneState
from stage_vla_v7.language import (
    DeterministicLanguageProvider,
    LanguageProviderRegistry,
)
from stage_vla_v7.vision import StaticVisionProvider, VisionProviderRegistry


def test_provider_registry_is_explicit_and_replace_safe() -> None:
    registry = ProviderRegistry[str]("vision")
    registry.register("demo", lambda config: str(config["value"]))
    assert registry.create("demo", {"value": 3}) == "3"
    with pytest.raises(ValueError, match="already registered"):
        registry.register("demo", lambda _config: "other")


def test_vision_registry_creates_only_vision_providers() -> None:
    registry = VisionProviderRegistry()
    scene = SceneState((ObjectDetection("red_cube", (0.4, 0.0, 0.02)),))
    registry.register("static", lambda _config: StaticVisionProvider(scene))
    registry.register("invalid", lambda _config: object())  # type: ignore[arg-type]

    assert isinstance(registry.create("static"), StaticVisionProvider)
    with pytest.raises(TypeError, match="VisionProvider"):
        registry.create("invalid")
    with pytest.raises(LookupError, match="unknown vision provider"):
        registry.create("missing")


def test_language_registry_creates_only_language_providers() -> None:
    registry = LanguageProviderRegistry()
    registry.register("deterministic", lambda _config: DeterministicLanguageProvider())
    registry.register("invalid", lambda _config: object())  # type: ignore[arg-type]

    assert isinstance(registry.create("deterministic"), DeterministicLanguageProvider)
    with pytest.raises(TypeError, match="LanguageProvider"):
        registry.create("invalid")
