"""Framework-neutral datasets for per-skill parameter learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable, Sequence

from stage_vla_v7.interfaces import RobotAction, Skill


@dataclass(frozen=True)
class TrainingSample:
    skill: Skill
    observation: tuple[float, ...]
    action: RobotAction

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.observation)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("training observation must contain finite values")
        object.__setattr__(self, "observation", values)


class ActionDataset(Sequence[TrainingSample]):
    """Validated in-memory samples for exactly one skill and observation schema."""

    def __init__(self, samples: Iterable[TrainingSample]) -> None:
        self._samples = tuple(samples)
        if not self._samples:
            raise ValueError("action dataset must contain at least one sample")
        skills = {sample.skill for sample in self._samples}
        dimensions = {len(sample.observation) for sample in self._samples}
        if len(skills) != 1 or len(dimensions) != 1:
            raise ValueError("action dataset must contain one skill and one observation dimension")

    @property
    def skill(self) -> Skill:
        return self._samples[0].skill

    @property
    def observation_dim(self) -> int:
        return len(self._samples[0].observation)

    def __getitem__(self, index: int) -> TrainingSample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)
