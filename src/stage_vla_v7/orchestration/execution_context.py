"""Application input kept separate from model and simulator implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stage_vla_v7.interfaces import VisionRequest


@dataclass(frozen=True)
class ExecutionContext:
    command: str
    frame: VisionRequest
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("execution command must be non-empty")
