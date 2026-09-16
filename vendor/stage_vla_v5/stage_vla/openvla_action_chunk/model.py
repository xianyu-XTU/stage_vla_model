"""Small continuous action head for frozen OpenVLA multimodal features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActionChunkConfig:
    feature_dim: int = 4096
    state_dim: int = 57
    chunk_length: int = 20
    state_hidden_dim: int = 128
    hidden_dim: int = 512
    dropout: float = 0.1
    action_limit: float = 1.0


class ActionChunkHead(nn.Module):
    """Predict a continuous 7-D action chunk from VLM features and robot state.

    The first six action dimensions are regressed in normalized units.  The
    gripper is predicted with one binary logit per future step and decoded to
    the environment-native -1 (close) / +1 (open) convention.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 4096,
        state_dim: int = 57,
        chunk_length: int = 20,
        state_hidden_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        action_limit: float = 1.0,
        arm_mean: np.ndarray | torch.Tensor | None = None,
        arm_std: np.ndarray | torch.Tensor | None = None,
        state_mean: np.ndarray | torch.Tensor | None = None,
        state_std: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or state_dim < 1 or chunk_length not in (10, 20):
            raise ValueError("feature/state dimensions must be positive and chunk_length must be 10 or 20")
        self.config = ActionChunkConfig(
            feature_dim=feature_dim,
            state_dim=state_dim,
            chunk_length=chunk_length,
            state_hidden_dim=state_hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            action_limit=action_limit,
        )

        self.register_buffer("arm_mean", self._stat(arm_mean, 6, 0.0))
        self.register_buffer("arm_std", self._stat(arm_std, 6, 1.0, minimum=1e-6))
        self.register_buffer("state_mean", self._stat(state_mean, state_dim, 0.0))
        self.register_buffer("state_std", self._stat(state_std, state_dim, 1.0, minimum=1e-6))

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(state_hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + state_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.arm_head = nn.Linear(hidden_dim, chunk_length * 6)
        self.gripper_head = nn.Linear(hidden_dim, chunk_length)

    @staticmethod
    def _stat(value, size: int, default: float, minimum: float | None = None) -> torch.Tensor:
        if value is None:
            tensor = torch.full((size,), default, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.shape != (size,) or not torch.isfinite(tensor).all():
            raise ValueError(f"normalization statistic must be finite [{size}], got {tuple(tensor.shape)}")
        if minimum is not None:
            tensor = tensor.clamp_min(minimum)
        return tensor

    @property
    def chunk_length(self) -> int:
        return self.config.chunk_length

    def forward(self, features: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(f"features must be [B,{self.config.feature_dim}], got {tuple(features.shape)}")
        if states.ndim != 2 or states.shape != (features.shape[0], self.config.state_dim):
            raise ValueError(f"states must be [B,{self.config.state_dim}], got {tuple(states.shape)}")
        normalized_state = (states - self.state_mean) / self.state_std
        fused = self.fusion(
            torch.cat((self.feature_norm(features.float()), self.state_encoder(normalized_state)), dim=-1)
        )
        arm_normalized = self.arm_head(fused).view(-1, self.chunk_length, 6)
        gripper_logits = self.gripper_head(fused).view(-1, self.chunk_length)
        return arm_normalized, gripper_logits

    def decode(self, arm_normalized: torch.Tensor, gripper_logits: torch.Tensor) -> torch.Tensor:
        if arm_normalized.shape[-2:] != (self.chunk_length, 6):
            raise ValueError("arm_normalized has an incompatible action-chunk shape")
        if gripper_logits.shape != arm_normalized.shape[:-1]:
            raise ValueError("gripper_logits must match the arm batch/chunk dimensions")
        arm = arm_normalized * self.arm_std + self.arm_mean
        arm = arm.clamp(-self.config.action_limit, self.config.action_limit)
        gripper = torch.where(gripper_logits >= 0, 1.0, -1.0).unsqueeze(-1)
        return torch.cat((arm, gripper), dim=-1)

    def act(self, features: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return self.decode(*self(features, states))

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.config)


def action_chunk_loss(
    arm_normalized: torch.Tensor,
    gripper_logits: torch.Tensor,
    target_actions: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    arm_mean: torch.Tensor,
    arm_std: torch.Tensor,
    gripper_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Masked expert-only loss; padded tail steps never contribute."""
    if target_actions.shape != (*arm_normalized.shape[:-1], 7):
        raise ValueError("target_actions must be [B,chunk,7]")
    if valid_mask.shape != arm_normalized.shape[:-1]:
        raise ValueError("valid_mask must be [B,chunk]")
    mask = valid_mask.to(dtype=arm_normalized.dtype)
    valid_steps = mask.sum()
    if valid_steps.item() <= 0:
        raise ValueError("action chunk contains no valid target steps")

    target_arm = (target_actions[..., :6] - arm_mean) / arm_std
    arm_per_dim = F.smooth_l1_loss(arm_normalized, target_arm, reduction="none")
    arm_loss = (arm_per_dim * mask.unsqueeze(-1)).sum() / (valid_steps * 6.0)

    target_gripper_open = (target_actions[..., 6] > 0).to(dtype=gripper_logits.dtype)
    gripper_per_step = F.binary_cross_entropy_with_logits(
        gripper_logits, target_gripper_open, reduction="none"
    )
    gripper_loss = (gripper_per_step * mask).sum() / valid_steps
    total = arm_loss + float(gripper_weight) * gripper_loss
    return {"total": total, "arm": arm_loss, "gripper": gripper_loss}
