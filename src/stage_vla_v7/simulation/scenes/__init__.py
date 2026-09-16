"""Scene composition and registration."""

from .base import SimulationScene
from .registry import SceneRegistry, default_scene_registry
from .stack_scene import SceneLayout, SceneManifest, StackScene

__all__ = [
    "SceneLayout",
    "SceneManifest",
    "SceneRegistry",
    "SimulationScene",
    "StackScene",
    "default_scene_registry",
]
