"""Protocol for application-owned Stage VLA pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from .action_interface import ActionResult
from .contracts import SkillToken
from .vision_interface import VisionRequest


PreparedT = TypeVar("PreparedT")


class PipelineInterface(Protocol[PreparedT]):
    def prepare(self, command: str, frame: VisionRequest) -> PreparedT: ...

    def act(
        self,
        prepared: PreparedT,
        token: SkillToken,
        observation: Sequence[float],
        *,
        finished: bool = False,
    ) -> ActionResult: ...
