"""Replaceable deterministic language boundary for V5.

The current implementation is intentionally instruction-template based.  A
future LLM adapter can return the same :class:`InstructionPlan` and leave the
task compiler, object-property provider and action models unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .task_dsl import InstructionParser, InstructionPlan


@dataclass(frozen=True)
class InstructionRequest:
    """Text-only request accepted by the deterministic language adapter."""

    command: str

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be a non-empty string")


class DeterministicInstructionAdapter:
    """Convert instruction-like text into typed tasks and skill tokens."""

    def __init__(self, parser: InstructionParser | None = None) -> None:
        self.parser = parser or InstructionParser()

    def predict(self, request: InstructionRequest | str) -> InstructionPlan:
        if isinstance(request, str):
            command = request
        elif isinstance(request, InstructionRequest):
            command = request.command
        else:
            raise TypeError("request must be InstructionRequest or str")
        return self.parser.plan(command)


__all__ = ["DeterministicInstructionAdapter", "InstructionRequest"]
