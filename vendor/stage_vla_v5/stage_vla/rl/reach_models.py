"""Checkpointable REACH action models separated from state measurement.

The structured policy consumes the same 52-D observation as learned REACH
actors.  It converts the already encoded fingertip-to-grasp-target error into
a bounded Cartesian action inside the checkpoint itself; no evaluator-side
teacher or recovery controller participates at runtime.
"""

from __future__ import annotations

import torch
from torch import nn

from .reach_policy import REACH_OBS_DIM


LEFT_TIP_ERROR_SLICE = slice(6, 9)
RIGHT_TIP_ERROR_SLICE = slice(9, 12)


class StructuredReachPolicy(nn.Module):
    """Deterministic proportional REACH head over the stable observation ABI."""

    def __init__(
        self,
        *,
        tip_error_normalization_m: float = 0.1,
        translation_limit_m: float = 0.005,
        pregrasp_tip_height_m: float = 0.01,
    ) -> None:
        super().__init__()
        if min(
            float(tip_error_normalization_m),
            float(translation_limit_m),
            float(pregrasp_tip_height_m),
        ) <= 0.0:
            raise ValueError("REACH geometry scales must be positive")
        self.tip_to_action_scale = float(
            tip_error_normalization_m / translation_limit_m
        )
        self.height_action_bias = float(
            pregrasp_tip_height_m / translation_limit_m
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2 or observation.size(1) != 52:
            raise RuntimeError("REACH observation must have shape [N,52]")
        tip_error = 0.5 * (
            observation[:, 6:9] + observation[:, 9:12]
        )
        translation = -self.tip_to_action_scale * tip_error
        height_bias = torch.zeros_like(translation)
        height_bias[:, 2] = self.height_action_bias
        translation = torch.clamp(translation + height_bias, -1.0, 1.0)
        yaw = torch.zeros(
            (observation.size(0), 1),
            dtype=observation.dtype,
            device=observation.device,
        )
        grip = torch.ones_like(yaw)
        return torch.cat((translation, yaw, grip), dim=1)


def build_structured_reach_policy() -> StructuredReachPolicy:
    """Construct the v5 policy using the frozen REACH observation/action scales."""
    return StructuredReachPolicy()


def build_reach_mlp() -> nn.Sequential:
    """Construct the shared learned REACH architecture in one canonical module."""
    return nn.Sequential(
        nn.Linear(REACH_OBS_DIM, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 64), nn.ELU(),
        nn.Linear(64, 5), nn.Tanh(),
    )


class GuardedReachPolicy(nn.Module):
    """Keep a learned path action unless it diverges from target geometry.

    The guard is part of the exported policy checkpoint.  It uses cosine
    agreement only; it does not invoke evaluator reference actions and cannot
    take over after a timeout.
    """

    def __init__(self, learned_policy: nn.Module, *, minimum_cosine: float = 0.90) -> None:
        super().__init__()
        if not -1.0 <= float(minimum_cosine) <= 1.0:
            raise ValueError("minimum_cosine must be in [-1,1]")
        self.learned_policy = learned_policy
        self.structured_policy = StructuredReachPolicy()
        self.minimum_cosine = float(minimum_cosine)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        learned = self.learned_policy(observation)
        structured = self.structured_policy(observation)
        learned_translation = learned[:, :3]
        structured_translation = structured[:, :3]
        dot = torch.sum(learned_translation * structured_translation, dim=1)
        denominator = torch.clamp(
            torch.linalg.vector_norm(learned_translation, dim=1)
            * torch.linalg.vector_norm(structured_translation, dim=1),
            min=1.0e-6,
        )
        use_learned = (dot / denominator) >= self.minimum_cosine
        translation = torch.where(
            use_learned.unsqueeze(1), learned_translation, structured_translation
        )
        yaw = torch.where(
            use_learned.unsqueeze(1), learned[:, 3:4], structured[:, 3:4]
        )
        grip = torch.ones_like(yaw)
        return torch.clamp(torch.cat((translation, yaw, grip), dim=1), -1.0, 1.0)
