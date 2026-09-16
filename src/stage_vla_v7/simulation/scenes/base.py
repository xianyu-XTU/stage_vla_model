"""Protocol implemented by registered simulation scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from stage_vla_v7.interfaces import SimulationModelDescriptor

if TYPE_CHECKING:
    from .stack_scene import SceneLayout, SceneManifest


@runtime_checkable
class SimulationScene(Protocol):
    @property
    def descriptor(self) -> SimulationModelDescriptor: ...

    def load(self) -> SceneManifest: ...

    def reset(self, seed: int | None = None) -> SceneLayout: ...
