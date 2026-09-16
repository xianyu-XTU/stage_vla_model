"""Small injected BC trainer for independent five-parameter skill policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import ActionDataset
from .trainer import TrainingAlgorithm, TrainingResult


@dataclass(frozen=True)
class BehaviorCloningConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 128)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("invalid behavior-cloning configuration")
        if not self.hidden_dims or any(value < 1 for value in self.hidden_dims):
            raise ValueError("hidden dimensions must be positive")


class BehaviorCloningTrainer:
    """Train a new optional MLP without changing migrated V5 checkpoints."""

    algorithm = TrainingAlgorithm.BC

    def __init__(self, config: BehaviorCloningConfig | None = None) -> None:
        self.config = config or BehaviorCloningConfig()

    def train(self, dataset: ActionDataset, output: Path) -> TrainingResult:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("behavior cloning requires the 'torch' extra") from exc
        cfg = self.config
        torch.manual_seed(cfg.seed)
        layers: list[object] = []
        previous = dataset.observation_dim
        for width in cfg.hidden_dims:
            layers.extend((torch.nn.Linear(previous, width), torch.nn.ReLU()))
            previous = width
        layers.extend((torch.nn.Linear(previous, 5), torch.nn.Tanh()))
        model = torch.nn.Sequential(*layers)
        observations = torch.tensor(
            [sample.observation for sample in dataset],
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [sample.action.values for sample in dataset],
            dtype=torch.float32,
        )
        batches = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(observations, actions),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        final_loss = 0.0
        model.train()
        for _epoch in range(cfg.epochs):
            for batch_observation, batch_action in batches:
                loss = torch.nn.functional.mse_loss(model(batch_observation), batch_action)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        torch.jit.script(model).save(str(output))
        return TrainingResult(dataset.skill, self.algorithm, output, {"final_mse": final_loss})
