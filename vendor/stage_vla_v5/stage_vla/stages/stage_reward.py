"""M8 stage-aware dense reward primitives.

Core shaping follows StARe's potential-based form:

    r'_t = r_t + gamma * Phi_{k_{t+1}}(s_{t+1}) - Phi_{k_t}(s_t)

and its geometric sigmoid potentials.  This project adds one explicit LIFT
stage (already verified in M6) and a small one-time bonus when a causal stage
boundary is crossed.  The transition bonus is a project engineering choice,
not a claim about the StARe paper's exact reward.

This module is pure PyTorch and does not import Isaac Lab.  It also does not
register itself as multiple RewardManager terms: the stateful previous-potential
update must happen exactly once per environment step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .stage_progress import ManipulationStage


@dataclass(frozen=True)
class StagePotentialConfig:
    object_size_m: float
    lift_scale_m: float

    def validate(self) -> None:
        if self.object_size_m <= 0:
            raise ValueError("object_size_m must be > 0")
        if self.lift_scale_m <= 0:
            raise ValueError("lift_scale_m must be > 0")


@dataclass(frozen=True)
class StageAwareRewardConfig:
    gamma: float
    stage_transition_bonus: float
    success_bonus: float

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        if self.stage_transition_bonus < 0:
            raise ValueError("stage_transition_bonus must be >= 0")
        if self.success_bonus < 0:
            raise ValueError("success_bonus must be >= 0")


@dataclass(frozen=True)
class StagePotentialInputs:
    reach_distance_m: Tensor
    grasp_error_m: Tensor
    lift_residual_m: Tensor
    transport_distance_m: Tensor
    place_distance_m: Tensor
    transport_scale_m: Tensor


@dataclass(frozen=True)
class StageAwareRewardDiagnostics:
    previous_potential: Tensor
    potential: Tensor
    shaping_reward: Tensor
    transition_bonus: Tensor
    sparse_success_reward: Tensor
    total_reward: Tensor


def normalized_progress_potential(distance_m: Tensor, scale_m: Tensor | float) -> Tensor:
    """StARe-style sigmoid potential ``sigmoid(1 - distance / scale)``."""
    distance = torch.as_tensor(distance_m)
    if not distance.is_floating_point():
        distance = distance.to(torch.float32)
    scale = torch.as_tensor(scale_m, device=distance.device, dtype=distance.dtype)
    if not torch.isfinite(distance).all() or not torch.isfinite(scale).all():
        raise ValueError("potential inputs contain NaN/Inf")
    if torch.any(distance < 0):
        raise ValueError("distance_m must be non-negative")
    if torch.any(scale <= 0):
        raise ValueError("scale_m must be > 0")
    return torch.sigmoid(1.0 - distance / scale)


def _same_shape(name: str, value: Tensor, expected: torch.Size, device) -> Tensor:
    out = torch.as_tensor(value, device=device)
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.shape != expected:
        raise ValueError(f"{name} shape {tuple(out.shape)} != {tuple(expected)}")
    return out


def stage_potential(
    stage: Tensor,
    inputs: StagePotentialInputs,
    *,
    cfg: StagePotentialConfig,
) -> Tensor:
    """Evaluate the active-stage potential for every environment."""
    cfg.validate()
    stage_t = torch.as_tensor(stage)
    if stage_t.dtype is not torch.long:
        raise TypeError(f"stage must be torch.long, got {stage_t.dtype}")
    expected = stage_t.shape
    device = stage_t.device

    reach = _same_shape("reach_distance_m", inputs.reach_distance_m, expected, device)
    grasp = _same_shape("grasp_error_m", inputs.grasp_error_m, expected, device)
    lift = _same_shape("lift_residual_m", inputs.lift_residual_m, expected, device)
    transport = _same_shape(
        "transport_distance_m", inputs.transport_distance_m, expected, device
    )
    place = _same_shape("place_distance_m", inputs.place_distance_m, expected, device)
    transport_scale = _same_shape(
        "transport_scale_m", inputs.transport_scale_m, expected, device
    )

    object_scale = torch.full_like(reach, float(cfg.object_size_m))
    lift_scale = torch.full_like(reach, float(cfg.lift_scale_m))

    reach_phi = normalized_progress_potential(reach, object_scale)
    grasp_phi = normalized_progress_potential(grasp, object_scale)
    lift_phi = normalized_progress_potential(lift, lift_scale)
    transport_phi = normalized_progress_potential(transport, transport_scale)
    place_phi = normalized_progress_potential(place, object_scale)

    valid = (stage_t >= int(ManipulationStage.REACH)) & (
        stage_t <= int(ManipulationStage.PLACE)
    )
    if not bool(valid.all().item()):
        raise ValueError(f"unknown stage ids: {stage_t[~valid].tolist()}")

    phi = torch.where(stage_t == int(ManipulationStage.REACH), reach_phi, grasp_phi)
    phi = torch.where(stage_t == int(ManipulationStage.LIFT), lift_phi, phi)
    phi = torch.where(stage_t == int(ManipulationStage.TRANSPORT), transport_phi, phi)
    phi = torch.where(stage_t == int(ManipulationStage.PLACE), place_phi, phi)
    return phi


class StageAwareRewardTracker:
    """Stateful previous-potential tracker; call exactly once per env step."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: StageAwareRewardConfig,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._previous_potential = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def previous_potential(self) -> Tensor:
        return self._previous_potential.clone()

    @property
    def initialized(self) -> Tensor:
        return self._initialized.clone()

    def reset(
        self,
        initial_potential: Tensor,
        env_ids: Sequence[int] | Tensor | None = None,
    ) -> None:
        phi = torch.as_tensor(initial_potential, device=self.device, dtype=torch.float32)
        if not torch.isfinite(phi).all():
            raise ValueError("initial_potential contains NaN/Inf")

        if env_ids is None:
            if phi.shape != (self.num_envs,):
                raise ValueError(
                    f"initial_potential must have shape ({self.num_envs},), got {tuple(phi.shape)}"
                )
            self._previous_potential.copy_(phi)
            self._initialized.fill_(True)
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be 1-D")
        if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= self.num_envs)):
            raise IndexError("env_ids out of range")
        if phi.shape == (self.num_envs,):
            phi_selected = phi[ids]
        elif phi.shape == ids.shape:
            phi_selected = phi
        else:
            raise ValueError(
                "initial_potential must be full-batch or match env_ids length"
            )
        if ids.numel():
            self._previous_potential[ids] = phi_selected
            self._initialized[ids] = True

    def update(
        self,
        current_potential: Tensor,
        transition_count: Tensor,
        just_succeeded: Tensor,
    ) -> StageAwareRewardDiagnostics:
        phi = torch.as_tensor(current_potential, device=self.device, dtype=torch.float32)
        transitions = torch.as_tensor(transition_count, device=self.device)
        succeeded = torch.as_tensor(just_succeeded, device=self.device)

        for name, value in (
            ("current_potential", phi),
            ("transition_count", transitions),
            ("just_succeeded", succeeded),
        ):
            if value.shape != (self.num_envs,):
                raise ValueError(
                    f"{name} must have shape ({self.num_envs},), got {tuple(value.shape)}"
                )
        if transitions.dtype is not torch.long:
            raise TypeError("transition_count must be torch.long")
        if succeeded.dtype is not torch.bool:
            raise TypeError("just_succeeded must be torch.bool")
        if torch.any(transitions < 0):
            raise ValueError("transition_count must be non-negative")
        if not torch.isfinite(phi).all():
            raise ValueError("current_potential contains NaN/Inf")
        if not bool(self._initialized.all().item()):
            missing = (~self._initialized).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"reward tracker not initialized for envs {missing}")

        previous = self._previous_potential.clone()
        shaping = self.cfg.gamma * phi - previous
        transition_bonus = transitions.to(phi.dtype) * self.cfg.stage_transition_bonus
        success_reward = succeeded.to(phi.dtype) * self.cfg.success_bonus
        total = shaping + transition_bonus + success_reward

        self._previous_potential.copy_(phi)

        return StageAwareRewardDiagnostics(
            previous_potential=previous,
            potential=phi.clone(),
            shaping_reward=shaping,
            transition_bonus=transition_bonus,
            sparse_success_reward=success_reward,
            total_reward=total,
        )
