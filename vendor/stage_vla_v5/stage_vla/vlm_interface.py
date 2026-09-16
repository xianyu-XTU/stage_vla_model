"""Replaceable interface for future vision-language models.

An implementation may use an image, RGB-D frame, language command or any
combination, but its output is restricted to a task spec and scene state.  It
cannot provide continuous robot actions or bypass the deterministic compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .task_dsl import TaskSpec
from .vision import Detection, SceneState, VisionStateAdapter


@dataclass(frozen=True)
class VLMRequest:
    """Model input container; payloads are opaque to the action runtime."""

    command: str | None = None
    image: object | None = None
    depth: object | None = None


@dataclass(frozen=True)
class VLMOutput:
    """Only structured semantics cross the VLM boundary."""

    task: TaskSpec
    detections: tuple[Detection, ...]
    held_label: str | None = None

    def scene_state(self, adapter: VisionStateAdapter | None = None) -> SceneState:
        return (adapter or VisionStateAdapter()).from_detections(
            self.detections, held_label=self.held_label
        )


class VLMAdapter(Protocol):
    """Protocol implemented by a future VLM or a deterministic test adapter."""

    def predict(self, request: VLMRequest) -> VLMOutput:
        """Return structured task and scene semantics, never an action."""


class StaticVLMAdapter:
    """Small dependency-free adapter useful for integration tests."""

    def __init__(self, task: TaskSpec, detections: Sequence[Detection], *, held_label: str | None = None) -> None:
        self._output = VLMOutput(task, tuple(detections), held_label)

    def predict(self, request: VLMRequest) -> VLMOutput:
        del request
        return self._output
