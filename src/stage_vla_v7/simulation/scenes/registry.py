"""Fail-closed scene registry."""

from __future__ import annotations

from .base import SimulationScene
from .stack_scene import StackScene


class SceneRegistry:
    def __init__(self) -> None:
        self._scenes: dict[str, SimulationScene] = {}

    def register(
        self,
        name: str,
        scene: SimulationScene,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(scene, SimulationScene):
            raise TypeError("scene must implement SimulationScene")
        if scene.descriptor.model_type != "scene":
            raise ValueError("scene descriptor must use model_type 'scene'")
        if name in self._scenes and not replace:
            raise ValueError(f"scene {name!r} is already registered")
        self._scenes[name] = scene

    def require(self, name: str) -> SimulationScene:
        try:
            return self._scenes[name]
        except KeyError as exc:
            raise LookupError(f"unknown scene {name!r}") from exc


def default_scene_registry() -> SceneRegistry:
    registry = SceneRegistry()
    registry.register("stack", StackScene())
    return registry
