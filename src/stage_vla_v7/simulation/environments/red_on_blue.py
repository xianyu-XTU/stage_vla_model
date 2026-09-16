"""Formal red-cube-on-blue-cube environment boundary."""

from __future__ import annotations

from stage_vla_v7.interfaces import SimulationEnvironment, SimulationModelDescriptor

from ..scenes import SceneLayout, StackScene
from .base import BackendEnvironment


class RedOnBlueEnvironment(BackendEnvironment):
    descriptor = SimulationModelDescriptor(
        "red-on-blue",
        "environment",
        "1",
        "backend-neutral",
        ("stack", "rgbd", "seeded-reset"),
    )
    command = "把红色方块放到蓝色方块上"
    object_label = "red_cube"
    support_label = "blue_cube"

    def __init__(
        self,
        scene: StackScene | None = None,
        backend: SimulationEnvironment | None = None,
    ) -> None:
        super().__init__(backend)
        self.scene = scene or StackScene()
        self.last_layout: SceneLayout | None = None

    def reset_layout(self, seed: int | None = None) -> SceneLayout:
        self.last_layout = self.scene.reset(seed)
        return self.last_layout

    def reset(self, seed: int | None = None):
        self.reset_layout(seed)
        return super().reset(seed)
