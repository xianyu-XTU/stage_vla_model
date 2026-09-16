"""M16 geometry-event micro-stage detector (shared train/inference).

Replaces the trace-phase labels with a single causal, geometry-event-driven
stage calculator used identically for BC training data, DAgger collection and
closed-loop inference. Includes hysteresis (a stage must persist N steps before
switching) and latching (RELEASE is terminal; a drop enters FAIL/REGRASP).

Micro-stages:
  ALIGN    red not yet aligned over blue (xy error above tolerance)
  DESCEND  xy aligned, red still above the stack height
  SETTLE   at stack height, still gripping, accumulating stable geometry frames
  RELEASE  release-ready after N stable frames; policy should command OPEN
  FAIL     cube dropped or unrecoverable (regrasp needed)

Order is monotonic ALIGN -> DESCEND -> SETTLE -> RELEASE; FAIL can occur from
any stage when the grasp is lost.
"""

from __future__ import annotations

import numpy as np
import torch

MICRO_NAMES = ("ALIGN", "DESCEND", "SETTLE", "RELEASE", "FAIL")
ALIGN, DESCEND, SETTLE, RELEASE, FAIL = range(5)


class GeometryPhaseConfig:
    def __init__(
        self,
        *,
        stack_height_diff_m: float = 0.0468,
        xy_align_tol_m: float = 0.04,
        z_descend_tol_m: float = 0.008,
        settle_speed_mps: float = 0.08,
        lift_height_m: float = 0.05,
        settle_steps: int = 3,
        hysteresis_steps: int = 3,
        release_latch: bool = True,
    ):
        self.stack_height_diff_m = stack_height_diff_m
        self.xy_align_tol_m = xy_align_tol_m
        self.z_descend_tol_m = z_descend_tol_m
        self.settle_speed_mps = settle_speed_mps
        self.lift_height_m = lift_height_m
        self.settle_steps = int(settle_steps)
        self.hysteresis_steps = int(hysteresis_steps)
        self.release_latch = release_latch


class GeometryPhaseTracker:
    """Vectorized causal phase tracker with hysteresis + latch."""

    def __init__(self, num_envs: int, cfg: GeometryPhaseConfig):
        self.cfg = cfg
        self._phase = torch.full((num_envs,), ALIGN, dtype=torch.long)
        self._pending_target = torch.full((num_envs,), ALIGN, dtype=torch.long)
        self._pending_count = torch.zeros(num_envs, dtype=torch.long)
        self._stable_count = torch.zeros(num_envs, dtype=torch.long)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._phase.fill_(ALIGN)
            self._pending_target.fill_(ALIGN)
            self._pending_count.zero_()
            self._stable_count.zero_()
        else:
            self._phase[env_ids] = ALIGN
            self._pending_target[env_ids] = ALIGN
            self._pending_count[env_ids] = 0
            self._stable_count[env_ids] = 0

    @property
    def phase(self) -> torch.Tensor:
        """Current latched phase (clone so callers cannot mutate tracker state)."""
        return self._phase.clone()

    def set_phase(self, env_ids, phase) -> None:
        """Initialize selected environments from a validated curriculum snapshot.

        M17-A4 can reset directly into DESCEND/SETTLE states captured from
        training demonstrations.  Priming the phase here prevents one ALIGN
        action from being emitted before the hysteresis machine re-discovers
        the snapshot's known micro-stage.
        """
        if env_ids is None:
            env_ids = torch.arange(len(self._phase), device=self._phase.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._phase.device)
        if torch.is_tensor(phase):
            values = phase.to(device=self._phase.device, dtype=torch.long)
            if values.ndim == 0:
                values = values.expand(len(env_ids))
        else:
            values = torch.full((len(env_ids),), int(phase), dtype=torch.long, device=self._phase.device)
        if values.shape != (len(env_ids),):
            raise ValueError(f"phase must be scalar or shape {(len(env_ids),)}, got {tuple(values.shape)}")
        if torch.any((values < ALIGN) | (values > FAIL)):
            raise ValueError("invalid geometry phase id")
        self._phase[env_ids] = values
        self._pending_target[env_ids] = values
        self._pending_count[env_ids] = 0
        self._stable_count[env_ids] = 0

    def update(
        self,
        red_pos: torch.Tensor,
        blue_pos: torch.Tensor,
        red_speed: torch.Tensor,
        gripper_open: torch.Tensor,
        physical_grasp: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        device = red_pos.device
        if self._phase.device != device:
            self._phase = self._phase.to(device)
            self._pending_target = self._pending_target.to(device)
            self._pending_count = self._pending_count.to(device)
            self._stable_count = self._stable_count.to(device)
        xy = torch.linalg.vector_norm(red_pos[..., :2] - blue_pos[..., :2], dim=-1)
        z_err = red_pos[..., 2] - (blue_pos[..., 2] + cfg.stack_height_diff_m)

        # target phase from geometry; all placement stages require the cube to be
        # LIFTED (a cube on the table is not in the placement micro-machine).
        lifted = red_pos[..., 2] > cfg.lift_height_m
        aligned = xy <= cfg.xy_align_tol_m
        descended = torch.abs(z_err) <= cfg.z_descend_tol_m  # band around stack height
        # SETTLE maintains a stable-count: N consecutive frames of aligned +
        # descended makes the stage release-READY. Entering RELEASE does NOT
        # require the gripper to be open yet -- opening is the action the BC must
        # output once it receives the RELEASE phase. NOTE: the low-speed gate was
        # dropped because the expert releases at varied cube speeds (0.07-0.34
        # m/s); velocity stays in the state for the policy to act on.
        settling = descended & aligned
        self._stable_count = torch.where(settling, self._stable_count + 1,
                                         torch.zeros_like(self._stable_count))
        stable_ready = self._stable_count >= cfg.settle_steps

        target = torch.full_like(self._phase, ALIGN)
        target = torch.where(lifted & aligned & ~descended, torch.tensor(DESCEND), target)
        target = torch.where(lifted & descended & ~stable_ready, torch.tensor(SETTLE), target)
        target = torch.where(lifted & stable_ready, torch.tensor(RELEASE), target)

        # grasp lost before entering (or while outside) RELEASE -> FAIL.
        # Once in RELEASE (latched), the grasp loss is a legitimate release.
        in_release = self._phase == RELEASE
        dropped = lifted & ~in_release & ~physical_grasp
        target = torch.where(dropped, torch.tensor(FAIL), target)

        # hysteresis (debounce): count consecutive steps the target has been a
        # fixed value; switch the phase only after it persists `hysteresis_steps`.
        same_as_pending = target == self._pending_target
        self._pending_count = torch.where(same_as_pending, self._pending_count + 1,
                                          torch.ones_like(self._pending_count))
        self._pending_target = target.clone()
        switch = (self._pending_count >= cfg.hysteresis_steps) & (target != self._phase)
        new_phase = torch.where(switch, target, self._phase)

        # latch: once RELEASE, never go back (except FAIL); FAIL is terminal
        if cfg.release_latch:
            was_release = self._phase == RELEASE
            new_phase = torch.where(was_release & (new_phase != FAIL),
                                    torch.tensor(RELEASE), new_phase)

        self._phase = new_phase
        return self._phase.clone()
