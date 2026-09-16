"""Explicit adapter metadata for the retained V5 StagePPO/DAgger runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from stage_vla_v7.interfaces import Skill

from .trainer import TrainingAlgorithm


@dataclass(frozen=True)
class LegacyTrainingCommand:
    skill: Skill
    algorithm: TrainingAlgorithm
    isaac_python: Path
    script: Path

    def build(self, arguments: Sequence[str]) -> tuple[str, ...]:
        if not self.isaac_python.is_file():
            raise FileNotFoundError(self.isaac_python)
        if not self.script.is_file():
            raise FileNotFoundError(self.script)
        return (str(self.isaac_python), str(self.script), "--skill", self.skill.value, *arguments)


def legacy_stageppo_command(
    repository_root: str | Path,
    *,
    skill: Skill | str,
    isaac_python: str | Path,
    algorithm: TrainingAlgorithm = TrainingAlgorithm.STAGE_PPO,
) -> LegacyTrainingCommand:
    if algorithm not in {TrainingAlgorithm.PPO, TrainingAlgorithm.STAGE_PPO, TrainingAlgorithm.DAGGER}:
        raise ValueError("the V5 adapter only serves PPO, StagePPO, and DAgger workflows")
    root = Path(repository_root).resolve()
    return LegacyTrainingCommand(
        Skill(skill),
        algorithm,
        Path(isaac_python).resolve(),
        root / "vendor" / "stage_vla_v5" / "tools" / "train_known_size_grasp.py",
    )
