"""Explicit training algorithm registry; orchestration never imports it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol

from stage_vla_v7.interfaces import Skill

from .dataset import ActionDataset


class TrainingAlgorithm(str, Enum):
    BC = "bc"
    DAGGER = "dagger"
    PPO = "ppo"
    STAGE_PPO = "stage_ppo"


@dataclass(frozen=True)
class TrainingResult:
    skill: Skill
    algorithm: TrainingAlgorithm
    checkpoint: Path
    metrics: Mapping[str, float]


class Trainer(Protocol):
    algorithm: TrainingAlgorithm

    def train(self, dataset: ActionDataset, output: Path) -> TrainingResult: ...


class TrainerRegistry:
    def __init__(self) -> None:
        self._trainers: dict[tuple[Skill, TrainingAlgorithm], Trainer] = {}

    def register(self, skill: Skill, trainer: Trainer, *, replace: bool = False) -> None:
        key = (skill, trainer.algorithm)
        if key in self._trainers and not replace:
            raise ValueError(f"trainer already registered for {skill.value}/{trainer.algorithm.value}")
        self._trainers[key] = trainer

    def require(self, skill: Skill | str, algorithm: TrainingAlgorithm | str) -> Trainer:
        key = (Skill(skill), TrainingAlgorithm(algorithm))
        try:
            return self._trainers[key]
        except KeyError as exc:
            raise LookupError(f"no trainer registered for {key[0].value}/{key[1].value}") from exc
