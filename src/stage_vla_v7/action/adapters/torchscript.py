"""Optional PyTorch adapter for exported per-skill policies."""

from __future__ import annotations

from pathlib import Path

from stage_vla_v7.contracts import ModelDescriptor, ProviderError, RobotAction, Skill

from ..interfaces import ActionRequest, ActionResult


class TorchScriptActionPolicy:
    """Run one V5/V7-compatible TorchScript policy without leaking Torch types."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        skill: Skill,
        observation_dim: int,
        device: str = "cpu",
        version: str = "1",
    ) -> None:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchScriptActionPolicy requires the 'torch' extra") from exc
        self._torch = torch
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.checkpoint = path
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
        values = torch.as_tensor(raw).detach().cpu().reshape(-1).tolist()
        if len(values) != 5:
            raise ProviderError("TorchScript policy did not return five action values")
        return ActionResult(
            RobotAction.from_values(values),
            self.descriptor,
            {"checkpoint": str(self.checkpoint)},
        )
