"""Train one optional V7 five-parameter skill policy from a JSON dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_vla_v7.action.training import (
    ActionDataset,
    BehaviorCloningConfig,
    BehaviorCloningTrainer,
    TrainingSample,
)
from stage_vla_v7.interfaces import RobotAction, Skill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True, choices=[skill.value for skill in Skill])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset JSON must be a non-empty list or contain a samples list")
    skill = Skill(args.skill)
    dataset = ActionDataset(
        TrainingSample(
            skill,
            tuple(row["observation"]),
            RobotAction.from_values(row["action"]),
        )
        for row in rows
    )
    trainer = BehaviorCloningTrainer(
        BehaviorCloningConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    )
    result = trainer.train(dataset, args.output)
    print(
        json.dumps(
            {
                "skill": result.skill.value,
                "algorithm": result.algorithm.value,
                "checkpoint": str(result.checkpoint),
                "metrics": dict(result.metrics),
            },
            indent=2,
        )
    )
