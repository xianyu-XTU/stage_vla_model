"""Pure-PyTorch M10 H2-S2 settling potential shaping.

M10-9-H2G proved that requiring strict ``settled(s_t)`` before the direct OPEN
credit is too sparse: the policy almost never observes base release-ready and
strictly settled while still grasping, so the OPEN learning signal disappears.

H2-S2 keeps M10-8-T1's causal direct-OPEN gate unchanged and adds a bounded
potential-difference reward for *approaching* the existing M7 settled velocity
boundary while the post-step state is base release-ready.  No success/stage/
action semantics are changed by this helper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


M10_H2S2_OPEN_GATE_SEMANTICS = "pre_action_release_ready"
M10_H2S2_SETTLE_SHAPING_SEMANTICS = "post_step_base_ready_normalized_speed_potential"


@dataclass(frozen=True)
class M10SettleShapingConfig:
    weight: float
    gamma: float
    speed_ratio_cap: float
    max_red_linear_speed_mps: float
    max_red_angular_speed_radps: float
    max_blue_linear_speed_mps: float
    max_blue_angular_speed_radps: float

    def validate(self) -> None:
        vals = torch.tensor(
            [
                self.weight,
                self.gamma,
                self.speed_ratio_cap,
                self.max_red_linear_speed_mps,
                self.max_red_angular_speed_radps,
                self.max_blue_linear_speed_mps,
                self.max_blue_angular_speed_radps,
            ],
            dtype=torch.float32,
        )
        if not torch.isfinite(vals).all():
            raise ValueError("settle-shaping parameters must be finite")
        if self.weight < 0:
            raise ValueError("weight must be >= 0")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must lie in [0, 1]")
        if self.speed_ratio_cap <= 1.0:
            raise ValueError("speed_ratio_cap must be > 1")
        for name, value in (
            ("max_red_linear_speed_mps", self.max_red_linear_speed_mps),
            ("max_red_angular_speed_radps", self.max_red_angular_speed_radps),
            ("max_blue_linear_speed_mps", self.max_blue_linear_speed_mps),
            ("max_blue_angular_speed_radps", self.max_blue_angular_speed_radps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0 for normalized settle shaping")


@dataclass(frozen=True)
class M10SettleShapingDiagnostics:
    release_ready: Tensor
    normalized_speed_ratio: Tensor
    settle_score: Tensor
    previous_potential: Tensor
    potential: Tensor
    shaping_reward: Tensor


def _as_speed(name: str, value: Tensor, *, device: torch.device, num_envs: int) -> Tensor:
    out = torch.as_tensor(value, dtype=torch.float32, device=device)
    if out.shape != (num_envs,):
        raise ValueError(f"{name} must have shape ({num_envs},), got {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if torch.any(out < 0):
        raise ValueError(f"{name} must be >= 0")
    return out


def normalized_settle_speed_ratio(
    red_linear_speed_mps: Tensor,
    red_angular_speed_radps: Tensor,
    blue_linear_speed_mps: Tensor,
    blue_angular_speed_radps: Tensor,
    *,
    cfg: M10SettleShapingConfig,
) -> Tensor:
    """Return the worst normalized speed relative to the strict M7 thresholds.

    Ratio <= 1 means all four speed gates satisfy M7 ``settled``.  Ratio > 1
    quantifies how far the worst component remains above its threshold.
    """

    cfg.validate()
    red_lin = torch.as_tensor(red_linear_speed_mps, dtype=torch.float32)
    device = red_lin.device
    if red_lin.ndim != 1:
        raise ValueError("speed tensors must be 1-D [num_envs]")
    n = int(red_lin.shape[0])
    red_lin = _as_speed("red_linear_speed_mps", red_lin, device=device, num_envs=n)
    red_ang = _as_speed(
        "red_angular_speed_radps", red_angular_speed_radps, device=device, num_envs=n
    )
    blue_lin = _as_speed(
        "blue_linear_speed_mps", blue_linear_speed_mps, device=device, num_envs=n
    )
    blue_ang = _as_speed(
        "blue_angular_speed_radps", blue_angular_speed_radps, device=device, num_envs=n
    )

    ratios = torch.stack(
        (
            red_lin / float(cfg.max_red_linear_speed_mps),
            red_ang / float(cfg.max_red_angular_speed_radps),
            blue_lin / float(cfg.max_blue_linear_speed_mps),
            blue_ang / float(cfg.max_blue_angular_speed_radps),
        ),
        dim=-1,
    )
    return ratios.amax(dim=-1)


def settle_score_from_ratio(ratio: Tensor, *, speed_ratio_cap: float) -> Tensor:
    """Map normalized speed ratio to [0, 1] with a dense pre-settled ramp.

    - ratio <= 1: score 1 (already inside the strict settled boundary)
    - ratio >= cap: score 0
    - between: linear ramp toward the strict boundary

    This intentionally rewards *approaching* settled instead of requiring the
    sparse binary condition to be true before any learning signal exists.
    """

    if speed_ratio_cap <= 1.0:
        raise ValueError("speed_ratio_cap must be > 1")
    r = torch.as_tensor(ratio, dtype=torch.float32)
    if not torch.isfinite(r).all():
        raise ValueError("ratio contains NaN/Inf")
    if torch.any(r < 0):
        raise ValueError("ratio must be >= 0")
    return torch.clamp(
        (float(speed_ratio_cap) - r) / (float(speed_ratio_cap) - 1.0),
        min=0.0,
        max=1.0,
    )


class M10SettleShapingTracker:
    """Stateful potential-difference settling reward.

    ``Phi_t`` is zero outside post-step base release-ready.  Inside base-ready,
    it is a bounded [0,1] score that rises as the worst normalized object speed
    approaches the existing M7 settled threshold.  Reward is:

        weight * (gamma * Phi_t - Phi_(t-1))

    so staying at the same score cannot farm positive reward indefinitely and
    losing placement readiness incurs the corresponding potential drop.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        cfg: M10SettleShapingConfig,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._previous_potential = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    @property
    def previous_potential(self) -> Tensor:
        return self._previous_potential.clone()

    def _ids(self, env_ids: Sequence[int] | Tensor | slice | None) -> Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("env_ids must be 1-D")
        if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= self.num_envs)):
            raise IndexError("env_ids out of range")
        return ids

    def reset(self, env_ids: Sequence[int] | Tensor | slice | None = None) -> None:
        ids = self._ids(env_ids)
        self._previous_potential[ids] = 0.0

    def update(
        self,
        release_ready: Tensor,
        red_linear_speed_mps: Tensor,
        red_angular_speed_radps: Tensor,
        blue_linear_speed_mps: Tensor,
        blue_angular_speed_radps: Tensor,
    ) -> M10SettleShapingDiagnostics:
        ready = torch.as_tensor(release_ready, device=self.device)
        if ready.shape != (self.num_envs,):
            raise ValueError(
                f"release_ready must have shape ({self.num_envs},), got {tuple(ready.shape)}"
            )
        if ready.dtype is not torch.bool:
            raise TypeError("release_ready must be torch.bool")

        ratio = normalized_settle_speed_ratio(
            torch.as_tensor(red_linear_speed_mps, device=self.device),
            torch.as_tensor(red_angular_speed_radps, device=self.device),
            torch.as_tensor(blue_linear_speed_mps, device=self.device),
            torch.as_tensor(blue_angular_speed_radps, device=self.device),
            cfg=self.cfg,
        ).to(self.device)
        score = settle_score_from_ratio(ratio, speed_ratio_cap=self.cfg.speed_ratio_cap).to(
            self.device
        )
        potential = torch.where(ready, score, torch.zeros_like(score))
        previous = self._previous_potential.clone()
        shaping = float(self.cfg.weight) * (float(self.cfg.gamma) * potential - previous)
        self._previous_potential.copy_(potential)

        return M10SettleShapingDiagnostics(
            release_ready=ready.clone(),
            normalized_speed_ratio=ratio.clone(),
            settle_score=score.clone(),
            previous_potential=previous,
            potential=potential.clone(),
            shaping_reward=shaping,
        )
