"""TorchScript parameter network exposed through the public ActionPolicy port."""

from __future__ import annotations

from pathlib import Path

from stage_vla_v7.interfaces import (
    ActionRequest,
    ActionResult,
    ModelDescriptor,
    ProviderError,
    RobotAction,
    Skill,
)

from .checkpoint_loader import load_torchscript_checkpoint
from .parameter_network import validate_action_parameters


class TorchScriptActionPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        skill: Skill,
        observation_dim: int,
        device: str = "cpu",
        version: str = "1",
        expected_sha256: str | None = None,
    ) -> None:
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        loaded = load_torchscript_checkpoint(
            checkpoint,
            device=device,
            expected_sha256=expected_sha256,
        )
        self._torch = __import__("torch")
        self.model = loaded.model
        self.checkpoint = loaded.path
        self.checkpoint_sha256 = loaded.sha256
        self.skill = skill
        self.observation_dim = observation_dim
        self.device = device
        self.descriptor = ModelDescriptor(
            name=f"torchscript-{skill.value.lower()}",
            version=version,
            kind="action",
            capabilities=("torchscript", "single-environment"),
        )

    def predict(self, request: ActionRequest) -> ActionResult:
        if request.skill is not self.skill:
            raise ProviderError(
                f"{self.descriptor.name} cannot serve skill {request.skill.value}"
            )
        torch = self._torch
        observation = torch.tensor(
            [request.observation], dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            raw = self.model(observation)
        try:
            values = validate_action_parameters(
                torch.as_tensor(raw).detach().cpu().reshape(-1).tolist()
            )
        except ValueError as exc:
            raise ProviderError(str(exc)) from exc
        return ActionResult(
            RobotAction.from_values(values),
            self.descriptor,
            {
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
            },
        )
