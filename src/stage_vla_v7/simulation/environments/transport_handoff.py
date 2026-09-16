"""TRANSPORT handoff thresholds and training-only settling reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class TransportHandoffConfig:
    success_xy_m: float = 0.045
    linear_speed_mps: float = 0.05
    angular_speed_radps: float = 1.0
    approach_width_m: float = 0.045
    linear_speed_weight: float = 1.25
    angular_speed_weight: float = 0.20
    action_weight: float = 0.20
    settled_bonus_weight: float = 1.50

    def validate(self) -> None:
        positive = (
            self.success_xy_m,
            self.linear_speed_mps,
            self.angular_speed_radps,
            self.approach_width_m,
        )
        non_negative = (
            self.linear_speed_weight,
            self.angular_speed_weight,
            self.action_weight,
            self.settled_bonus_weight,
        )
        if any(
            not torch.isfinite(torch.tensor(value)) or value <= 0
            for value in positive
        ):
            raise ValueError("TRANSPORT handoff thresholds must be finite and positive")
        if any(
            not torch.isfinite(torch.tensor(value)) or value < 0
            for value in non_negative
        ):
            raise ValueError(
                "TRANSPORT handoff reward weights must be finite and non-negative"
            )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def transport_handoff_ready(
    physical_grasp: torch.Tensor,
    pressure_ok: torch.Tensor,
    xy_distance_m: torch.Tensor,
    linear_speed_mps: torch.Tensor,
    angular_speed_radps: torch.Tensor,
    *,
    cfg: TransportHandoffConfig,
) -> torch.Tensor:
    cfg.validate()
    physical = torch.as_tensor(physical_grasp, dtype=torch.bool)
    pressure = torch.as_tensor(pressure_ok, device=physical.device, dtype=torch.bool)
    xy = torch.as_tensor(xy_distance_m, device=physical.device, dtype=torch.float32)
    linear = torch.as_tensor(
        linear_speed_mps, device=physical.device, dtype=torch.float32
    )
    angular = torch.as_tensor(
        angular_speed_radps, device=physical.device, dtype=torch.float32
    )
    if not (
        physical.shape == pressure.shape == xy.shape == linear.shape == angular.shape
    ):
        raise ValueError("TRANSPORT handoff inputs must have matching shapes")
    if (
        not torch.isfinite(xy).all()
        or not torch.isfinite(linear).all()
        or not torch.isfinite(angular).all()
    ):
        raise ValueError("TRANSPORT distances and speeds must be finite")
    return (
        physical
        & pressure
        & (xy <= float(cfg.success_xy_m))
        & (linear <= float(cfg.linear_speed_mps))
        & (angular <= float(cfg.angular_speed_radps))
    )


def transport_settle_reward_terms(
    xy_distance_m: torch.Tensor,
    linear_speed_mps: torch.Tensor,
    angular_speed_radps: torch.Tensor,
    normalized_action: torch.Tensor,
    *,
    cfg: TransportHandoffConfig,
) -> dict[str, torch.Tensor]:
    cfg.validate()
    xy = torch.as_tensor(xy_distance_m, dtype=torch.float32)
    linear = torch.as_tensor(linear_speed_mps, device=xy.device, dtype=xy.dtype)
    angular = torch.as_tensor(angular_speed_radps, device=xy.device, dtype=xy.dtype)
    action = torch.as_tensor(normalized_action, device=xy.device, dtype=xy.dtype)
    if xy.ndim != 1 or linear.shape != xy.shape or angular.shape != xy.shape:
        raise ValueError("TRANSPORT reward distances and speeds must have shape [N]")
    if action.shape != (xy.shape[0], 5):
        raise ValueError("normalized_action must have shape [N,5]")
    if not all(
        torch.isfinite(value).all() for value in (xy, linear, angular, action)
    ):
        raise ValueError("TRANSPORT reward inputs must be finite")
    approach = (
        1.0
        - (xy - float(cfg.success_xy_m)).clamp_min(0.0)
        / float(cfg.approach_width_m)
    ).clamp(0.0, 1.0)
    linear_ratio = (linear / float(cfg.linear_speed_mps)).clamp(0.0, 4.0)
    angular_ratio = (angular / float(cfg.angular_speed_radps)).clamp(0.0, 4.0)
    return {
        "approach": approach,
        "linear_speed_cost": approach
        * float(cfg.linear_speed_weight)
        * linear_ratio,
        "angular_speed_cost": approach
        * float(cfg.angular_speed_weight)
        * angular_ratio,
        "action_cost": approach
        * float(cfg.action_weight)
        * action[:, :2].square().sum(dim=-1),
        "settled_bonus": approach
        * float(cfg.settled_bonus_weight)
        * torch.exp(-linear_ratio)
        * torch.exp(-angular_ratio),
    }
