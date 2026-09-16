"""Pure-tensor terminal stability helpers for M17-A5.

A5 does not change the frozen PlaceBC or the PPO network.  It changes only the
terminal execution contract:

* RELEASE motion remains locked to zero (already true in M17).
* OPEN is delayed until the object has remained inside strict stack geometry
  and below a low-speed threshold for N consecutive RELEASE frames.
* Once OPEN has been issued it is latched for the rest of the episode.

The gate is intentionally independent of Isaac Lab so it can be unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stage_vla.lightweight_vla.geometry_phase import RELEASE


@dataclass(frozen=True)
class ReleaseStabilityConfig:
    gate_steps: int = 0
    speed_threshold_mps: float = 0.02
    xy_tolerance_m: float = 0.04
    z_tolerance_m: float = 0.010

    def validate(self) -> None:
        if self.gate_steps < 0:
            raise ValueError("gate_steps must be >= 0")
        if self.speed_threshold_mps <= 0:
            raise ValueError("speed_threshold_mps must be > 0")
        if self.xy_tolerance_m <= 0 or self.z_tolerance_m <= 0:
            raise ValueError("geometry tolerances must be > 0")


class ReleaseStabilityGate:
    """Vectorized, latched low-speed gate for the RELEASE -> OPEN event.

    ``gate_steps == 0`` preserves original M17 semantics: OPEN immediately when
    the geometry phase becomes RELEASE.
    """

    def __init__(self, num_envs: int, device: torch.device | str, cfg: ReleaseStabilityConfig):
        cfg.validate()
        self.cfg = cfg
        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._opened = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def count(self) -> torch.Tensor:
        return self._count.clone()

    @property
    def opened(self) -> torch.Tensor:
        return self._opened.clone()

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._count.zero_()
            self._opened.zero_()
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._count.device)
        self._count[env_ids] = 0
        self._opened[env_ids] = False

    def update(
        self,
        phase: torch.Tensor,
        xy_error_m: torch.Tensor,
        z_error_abs_m: torch.Tensor,
        speed_mps: torch.Tensor,
    ) -> torch.Tensor:
        phase = phase.to(self._count.device)
        xy_error_m = xy_error_m.to(self._count.device)
        z_error_abs_m = z_error_abs_m.to(self._count.device)
        speed_mps = speed_mps.to(self._count.device)

        in_release = phase == RELEASE
        if self.cfg.gate_steps == 0:
            self._opened |= in_release
            return self._opened.clone()

        eligible = (
            in_release
            & (xy_error_m <= self.cfg.xy_tolerance_m)
            & (z_error_abs_m <= self.cfg.z_tolerance_m)
            & (speed_mps <= self.cfg.speed_threshold_mps)
        )
        self._count = torch.where(
            self._opened,
            self._count,
            torch.where(eligible, self._count + 1, torch.zeros_like(self._count)),
        )
        trigger = eligible & (self._count >= self.cfg.gate_steps)
        self._opened |= trigger
        return self._opened.clone()
