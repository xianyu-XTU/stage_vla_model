"""Metadata for external training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stage_vla_v7.interfaces import Skill

from ..network import sha256_file


@dataclass(frozen=True)
class CheckpointRecord:
    logical_name: str
    path: Path
    sha256: str
    skill: Skill
    model_type: str
    version: str
    observation_dim: int
    action_dim: int = 5
    training_source: str = "unknown"

    def validate(self) -> None:
        resolved = self.path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if self.observation_dim < 1 or self.action_dim != 5:
            raise ValueError("checkpoint dimensions violate the V7 action contract")
        if sha256_file(resolved) != self.sha256.upper():
            raise ValueError(f"checkpoint SHA256 mismatch for {self.logical_name}")
