"""Dependency-free vision providers for tests and integration bring-up."""

from __future__ import annotations

from collections.abc import Callable

from stage_vla_v7.contracts import ModelDescriptor, SceneState

from ..interfaces import VisionRequest, VisionResult


class StaticVisionProvider:
    """Return a fixed scene while preserving request frame metadata."""

    descriptor = ModelDescriptor(
        name="static-vision",
        version="1",
        kind="vision",
        capabilities=("deterministic", "test"),
    )

    def __init__(self, scene: SceneState) -> None:
        self.scene = scene

    def detect(self, request: VisionRequest) -> VisionResult:
        scene = SceneState(
            detections=self.scene.detections,
            frame_id=request.frame_id or self.scene.frame_id,
            timestamp_s=(
                request.timestamp_s
                if request.timestamp_s is not None
                else self.scene.timestamp_s
            ),
            held_label=self.scene.held_label,
        )
        return VisionResult(scene, self.descriptor, {"static": True})


class CallableVisionProvider:
    """Adapt a callable without coupling core code to a framework SDK."""

    def __init__(
        self,
        function: Callable[[VisionRequest], SceneState],
        *,
        name: str,
        version: str = "1",
        capabilities: tuple[str, ...] = (),
    ) -> None:
        self.function = function
        self.descriptor = ModelDescriptor(name, version, "vision", capabilities)

    def detect(self, request: VisionRequest) -> VisionResult:
        scene = self.function(request)
        if not isinstance(scene, SceneState):
            raise TypeError("vision callable must return SceneState")
        return VisionResult(scene, self.descriptor)
