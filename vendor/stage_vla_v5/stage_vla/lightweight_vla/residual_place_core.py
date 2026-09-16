"""Pure tensor utilities for M17 PLACE Residual PPO.

This module intentionally has no Isaac Lab dependency so its action composition,
observation semantics and reward math can be unit-tested with ordinary Python.

The frozen M16 PlaceBC remains the base controller.  PPO predicts only a small
4-D residual over ``[dx, dy, dz, dyaw]``.  Gripper timing remains governed by
the causal GeometryPhaseTracker contract: CLOSE before RELEASE, OPEN in RELEASE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .geometry_phase import ALIGN, DESCEND, FAIL, RELEASE, SETTLE
from .place_bc import CONT_DIM, STATE_DIM

# Base BC state (25) + explicit task-centric derived features:
#   ee_rel_target xyz (3), xy_error (1), signed z_error (1), red_speed (1)
RESIDUAL_OBS_DIM = STATE_DIM + 6
# M16 BC state uses the nominal 0.040 m cube-center offset, while the strict
# evaluator uses the empirically calibrated 0.0468 m stacked center difference.
STRICT_STACK_HEIGHT_DIFF_M = 0.0468
BC_TARGET_HEIGHT_DIFF_M = 0.0400
STRICT_Z_BIAS_M = STRICT_STACK_HEIGHT_DIFF_M - BC_TARGET_HEIGHT_DIFF_M


@dataclass(frozen=True)
class ResidualRewardConfig:
    """Small, interpretable PLACE-only reward used by the residual policy."""

    xy_progress_weight: float = 40.0
    z_progress_weight: float = 30.0
    near_xy_weight: float = 0.20
    near_z_weight: float = 0.20
    settle_weight: float = 0.35
    phase_progress_weight: float = 0.25
    residual_l2_weight: float = 0.01
    success_bonus: float = 25.0
    fail_penalty: float = 10.0
    xy_std_m: float = 0.03
    z_std_m: float = 0.02
    settle_speed_std_mps: float = 0.05


def residual_observation_from_bc_state(state: torch.Tensor) -> torch.Tensor:
    """Augment the existing 25-D object-centric BC state for Residual PPO.

    The M16 state is already translation-invariant/object-centric.  Rather than
    replacing it with another equivalent representation, M17 makes the most
    important terminal errors explicit for the small residual actor.
    """
    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"expected BC state dim {STATE_DIM}, got {state.shape[-1]}")
    red_rel_target = state[..., 0:3]
    ee_rel_red = state[..., 3:6]
    ee_rel_target = red_rel_target + ee_rel_red
    xy_error = torch.linalg.vector_norm(red_rel_target[..., :2], dim=-1, keepdim=True)
    # Residual PPO gets the *strict-evaluator* z error explicitly. The frozen
    # BC still receives its original 25-D state unchanged.
    z_error = red_rel_target[..., 2:3] - STRICT_Z_BIAS_M
    red_speed = torch.linalg.vector_norm(state[..., 6:9], dim=-1, keepdim=True)
    out = torch.cat([state, ee_rel_target, xy_error, z_error, red_speed], dim=-1)
    if out.shape[-1] != RESIDUAL_OBS_DIM:
        raise RuntimeError("residual observation dimension invariant broken")
    return out


def stage_action_abs_quantile(
    cont: torch.Tensor,
    micro: torch.Tensor,
    *,
    q: float = 0.95,
    num_stages: int = 5,
) -> torch.Tensor:
    """Per-stage absolute action quantiles from expert/DAgger data.

    Missing stages receive zeros and should later be filled by a conservative
    minimum scale.  RELEASE/FAIL are explicitly zeroed by ``make_residual_scales``.
    """
    if cont.ndim != 2 or cont.shape[-1] != CONT_DIM:
        raise ValueError(f"cont must be [N,{CONT_DIM}]")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1]")
    rows = []
    for stage in range(num_stages):
        x = cont[micro == stage].abs()
        if len(x) == 0:
            rows.append(torch.zeros(CONT_DIM, dtype=cont.dtype, device=cont.device))
        else:
            rows.append(torch.quantile(x, q, dim=0))
    return torch.stack(rows, dim=0)


def make_residual_scales(
    p95: torch.Tensor,
    *,
    factor: float = 1.0,
    min_xyz: float = 0.002,
    min_yaw: float = 0.002,
    max_xyz: float = 0.03,
    max_yaw: float = 0.05,
    settle_multiplier: float = 1.0,
) -> torch.Tensor:
    """Turn expert per-stage action ranges into safe PPO residual ranges.

    The residual is deliberately bounded to the scale of demonstrated actions,
    never a new full controller. RELEASE and FAIL cannot move XYZ/yaw at all.
    SETTLE is additionally capped at half of the global max to prevent a late
    large correction from destroying an almost-successful stack. M17-A5 may
    further shrink only the SETTLE residual via ``settle_multiplier`` while
    leaving ALIGN/DESCEND behavior unchanged.
    """
    if p95.shape != (5, CONT_DIM):
        raise ValueError(f"expected p95 shape (5,{CONT_DIM}), got {tuple(p95.shape)}")
    if not (0.0 < float(settle_multiplier) <= 1.0):
        raise ValueError("settle_multiplier must be in (0,1]")
    floors = torch.tensor([min_xyz, min_xyz, min_xyz, min_yaw], dtype=p95.dtype, device=p95.device)
    caps = torch.tensor([max_xyz, max_xyz, max_xyz, max_yaw], dtype=p95.dtype, device=p95.device)
    scales = torch.maximum(p95 * float(factor), floors.unsqueeze(0)).minimum(caps.unsqueeze(0))
    # Terminal safety contract.
    settle_caps = torch.tensor(
        [max_xyz * 0.5, max_xyz * 0.5, max_xyz * 0.5, max_yaw * 0.5],
        dtype=p95.dtype,
        device=p95.device,
    )
    scales[SETTLE] = torch.minimum(scales[SETTLE], settle_caps) * float(settle_multiplier)
    scales[RELEASE].zero_()
    scales[FAIL].zero_()
    return scales


def compose_residual_cont_action(
    base_cont: torch.Tensor,
    residual_unit: torch.Tensor,
    phase: torch.Tensor,
    stage_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose frozen BC action + bounded phase-aware residual.

    Returns ``(final_cont, residual_delta)`` in the raw default-env action units.
    RELEASE is motion-locked: the policy only opens the gripper there.
    """
    if base_cont.shape != residual_unit.shape or base_cont.shape[-1] != CONT_DIM:
        raise ValueError("base_cont and residual_unit must have matching [...,4] shape")
    if stage_scales.shape != (5, CONT_DIM):
        raise ValueError("stage_scales must be [5,4]")
    residual_unit = residual_unit.clamp(-1.0, 1.0)
    scale = stage_scales.to(base_cont.device)[phase]
    delta = residual_unit * scale
    final = base_cont + delta
    release = phase == RELEASE
    fail = phase == FAIL
    final = torch.where((release | fail).unsqueeze(-1), torch.zeros_like(final), final)
    delta = torch.where((release | fail).unsqueeze(-1), torch.zeros_like(delta), delta)
    return final, delta


def strict_grip_command(phase: torch.Tensor) -> torch.Tensor:
    """Causal M16-D2.1 grip contract: OPEN iff phase is RELEASE."""
    return torch.where(
        phase == RELEASE,
        torch.ones_like(phase, dtype=torch.float32),
        -torch.ones_like(phase, dtype=torch.float32),
    )


def residual_place_reward(
    prev_bc_state: torch.Tensor,
    next_bc_state: torch.Tensor,
    prev_phase: torch.Tensor,
    next_phase: torch.Tensor,
    residual_unit: torch.Tensor,
    success: torch.Tensor,
    fail: torch.Tensor,
    *,
    cfg: ResidualRewardConfig | None = None,
) -> torch.Tensor:
    """Dense progress + sparse strict success reward for PLACE refinement."""
    cfg = cfg or ResidualRewardConfig()
    prev_err = prev_bc_state[..., 0:3]
    next_err = next_bc_state[..., 0:3]
    prev_xy = torch.linalg.vector_norm(prev_err[..., :2], dim=-1)
    next_xy = torch.linalg.vector_norm(next_err[..., :2], dim=-1)
    prev_z = (prev_err[..., 2] - STRICT_Z_BIAS_M).abs()
    next_z = (next_err[..., 2] - STRICT_Z_BIAS_M).abs()
    speed = torch.linalg.vector_norm(next_bc_state[..., 6:9], dim=-1)

    rew = cfg.xy_progress_weight * (prev_xy - next_xy)
    rew = rew + cfg.z_progress_weight * (prev_z - next_z)
    rew = rew + cfg.near_xy_weight * torch.exp(-next_xy / cfg.xy_std_m)
    rew = rew + cfg.near_z_weight * torch.exp(-next_z / cfg.z_std_m)

    near_stack = (next_xy < 0.04) & (next_z < 0.010)
    rew = rew + cfg.settle_weight * near_stack.float() * torch.exp(-speed / cfg.settle_speed_std_mps)

    # Reward monotonic useful transitions only; FAIL has numeric id 4 but is not progress.
    useful_progress = (
        ((prev_phase == ALIGN) & (next_phase == DESCEND))
        | ((prev_phase == DESCEND) & (next_phase == SETTLE))
        | ((prev_phase == SETTLE) & (next_phase == RELEASE))
    )
    rew = rew + cfg.phase_progress_weight * useful_progress.float()
    rew = rew - cfg.residual_l2_weight * residual_unit.pow(2).sum(dim=-1)
    rew = rew + cfg.success_bonus * success.float()
    rew = rew - cfg.fail_penalty * fail.float()
    return rew
