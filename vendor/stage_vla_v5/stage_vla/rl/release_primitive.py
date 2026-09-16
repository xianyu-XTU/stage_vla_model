"""M17-A6 release primitive: OPEN -> short hold -> controlled +Z retreat.

The controller is pure tensor logic and does not own physics.  It only decides
(1) whether the gripper should be OPEN/CLOSED in RELEASE and (2) whether a small
positive-Z Cartesian retreat should override the otherwise zero-locked RELEASE
motion.

Modes
-----
baseline
    Preserve original M17 semantics: OPEN immediately in RELEASE, no retreat.
retreat
    OPEN immediately in RELEASE.  Wait until the physical gripper is observed
    open, hold for ``open_hold_steps``, then retreat +Z for ``retreat_steps``.
support_retreat
    Require exact red<->blue support contact for ``support_gate_steps``
    consecutive RELEASE frames before commanding OPEN.  In v10.7 the default
    evaluation backend is the project's validated raw Omni PhysX body-pair
    report rather than a filtered ContactSensor. A bounded fallback can
    be enabled with ``support_max_wait_steps`` so a sensor/configuration miss
    cannot deadlock evaluation forever.  After physical OPEN, hold then retreat.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stage_vla.lightweight_vla.geometry_phase import RELEASE

BASELINE = "baseline"
RETREAT = "retreat"
SUPPORT_RETREAT = "support_retreat"
VALID_MODES = (BASELINE, RETREAT, SUPPORT_RETREAT)


@dataclass(frozen=True)
class ReleasePrimitiveConfig:
    mode: str = BASELINE
    open_hold_steps: int = 3
    retreat_steps: int = 8
    retreat_dz_per_step: float = 0.003
    support_gate_steps: int = 2
    support_force_threshold_n: float = 0.20
    support_max_wait_steps: int = 20

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"release primitive mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.open_hold_steps < 0 or self.retreat_steps < 0:
            raise ValueError("open_hold_steps and retreat_steps must be >= 0")
        if self.retreat_dz_per_step < 0:
            raise ValueError("retreat_dz_per_step must be >= 0")
        if self.support_gate_steps <= 0:
            raise ValueError("support_gate_steps must be > 0")
        if self.support_force_threshold_n < 0:
            raise ValueError("support_force_threshold_n must be >= 0")
        if self.support_max_wait_steps < 0:
            raise ValueError("support_max_wait_steps must be >= 0")


class ReleasePrimitiveController:
    """Vectorized latched release/retreat state machine."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg: ReleasePrimitiveConfig):
        cfg.validate()
        self.cfg = cfg
        self._release_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._support_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._open_commanded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._physical_open_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._retreat_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._fallback_open = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._release_age.zero_()
            self._support_count.zero_()
            self._open_commanded.zero_()
            self._physical_open_age.zero_()
            self._retreat_count.zero_()
            self._fallback_open.zero_()
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._release_age.device)
        for buf in (
            self._release_age,
            self._support_count,
            self._open_commanded,
            self._physical_open_age,
            self._retreat_count,
            self._fallback_open,
        ):
            if buf.dtype == torch.bool:
                buf[ids] = False
            else:
                buf[ids] = 0

    @property
    def open_commanded(self) -> torch.Tensor:
        return self._open_commanded.clone()

    @property
    def support_count(self) -> torch.Tensor:
        return self._support_count.clone()

    @property
    def retreat_count(self) -> torch.Tensor:
        return self._retreat_count.clone()

    @property
    def fallback_open(self) -> torch.Tensor:
        return self._fallback_open.clone()

    def update(
        self,
        phase: torch.Tensor,
        gripper_open: torch.Tensor,
        support_force_n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(grip_cmd, retreat_dz)`` for the current control step.

        ``grip_cmd`` uses the project's raw gripper convention: +1 OPEN, -1 CLOSE.
        ``retreat_dz`` is an additive raw relative Cartesian +Z command.
        """
        device = self._release_age.device
        phase = phase.to(device)
        gripper_open = gripper_open.to(device=device, dtype=torch.bool)
        support_force_n = support_force_n.to(device=device, dtype=torch.float32)
        in_release = phase == RELEASE

        # Leaving RELEASE only occurs on reset/FAIL in the current latched phase
        # machine. Keep non-release rows closed and clear transient counters.
        self._release_age = torch.where(in_release, self._release_age + 1, torch.zeros_like(self._release_age))
        self._support_count = torch.where(in_release, self._support_count, torch.zeros_like(self._support_count))
        self._physical_open_age = torch.where(in_release, self._physical_open_age, torch.zeros_like(self._physical_open_age))
        self._retreat_count = torch.where(in_release, self._retreat_count, torch.zeros_like(self._retreat_count))

        if self.cfg.mode == BASELINE:
            self._open_commanded |= in_release
        elif self.cfg.mode == RETREAT:
            self._open_commanded |= in_release
        else:
            support_now = in_release & (support_force_n >= self.cfg.support_force_threshold_n)
            self._support_count = torch.where(
                self._open_commanded,
                self._support_count,
                torch.where(support_now, self._support_count + 1, torch.zeros_like(self._support_count)),
            )
            contact_trigger = in_release & (self._support_count >= self.cfg.support_gate_steps)
            fallback_trigger = torch.zeros_like(contact_trigger)
            if self.cfg.support_max_wait_steps > 0:
                fallback_trigger = in_release & (self._release_age >= self.cfg.support_max_wait_steps)
            self._fallback_open |= fallback_trigger & ~contact_trigger & ~self._open_commanded
            self._open_commanded |= contact_trigger | fallback_trigger

        # OPEN is latched for the rest of a RELEASE episode.
        grip_cmd = torch.where(
            in_release & self._open_commanded,
            torch.ones_like(phase, dtype=torch.float32),
            -torch.ones_like(phase, dtype=torch.float32),
        )

        physically_open = in_release & self._open_commanded & gripper_open
        self._physical_open_age = torch.where(
            physically_open,
            self._physical_open_age + 1,
            torch.where(self._open_commanded, self._physical_open_age, torch.zeros_like(self._physical_open_age)),
        )

        retreat_active = (
            physically_open
            & (self._physical_open_age > self.cfg.open_hold_steps)
            & (self._retreat_count < self.cfg.retreat_steps)
            & (self.cfg.mode != BASELINE)
        )
        self._retreat_count = torch.where(retreat_active, self._retreat_count + 1, self._retreat_count)
        dz = torch.where(
            retreat_active,
            torch.full_like(support_force_n, float(self.cfg.retreat_dz_per_step)),
            torch.zeros_like(support_force_n),
        )
        return grip_cmd, dz
