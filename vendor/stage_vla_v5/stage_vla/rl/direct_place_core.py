"""M23 direct PPO action/reward contract, independent of Isaac and old policies.

There is no BC base action, residual composition, geometry phase, release gate,
or open latch. Five policy outputs own XYZ, yaw, and binary gripper decisions.
Roll/pitch remain fixed by the low-level pose IK action interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

ACTION_DIM = 5
OBS_DIM = 46
TRAIN_SOURCE_SEEDS = frozenset({1004, 1009, 1015, 1020, 1031})


@dataclass(frozen=True)
class DirectPlaceConfig:
    translation_limit_m: float = .002
    yaw_limit_rad: float = .01
    stack_height_m: float = .0468
    xy_success_m: float = .04
    z_success_m: float = .010
    speed_success_mps: float = .05
    stable_steps: int = 5
    episode_steps: int = 180
    fallen_height_diff_m: float = .025
    far_xy_m: float = .15
    failure_consecutive_steps: int = 2
    gamma: float = .98
    success_bonus: float = 25.
    failure_penalty: float = 10.
    step_cost: float = .01
    movement_cost: float = .001

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value):
                raise ValueError(f"nonfinite {name}")
        if not 0 < self.gamma <= 1:
            raise ValueError("invalid discount")
        if min(self.translation_limit_m, self.yaw_limit_rad, self.stable_steps,
               self.episode_steps, self.failure_consecutive_steps) <= 0:
            raise ValueError("action limits and horizons must be positive")
        if not 0 < self.fallen_height_diff_m < self.stack_height_m - self.z_success_m:
            raise ValueError("fall threshold overlaps success geometry")
        if min(self.step_cost, self.movement_cost, self.failure_penalty, self.success_bonus) < 0:
            raise ValueError("reward scales must be nonnegative")


def direct_action(policy_action, cfg=None):
    """Return metric pose delta + OPEN/CLOSE; backend arm scale MUST be 1.0.

    Sign threshold is actuator decoding, not a state-dependent decision. OPEN
    and CLOSE are always available, including reclosing after a prior OPEN.
    """
    cfg = cfg or DirectPlaceConfig()
    if policy_action.ndim != 2 or policy_action.shape[1] != ACTION_DIM:
        raise ValueError("direct policy action must be [N,5]")
    if not torch.isfinite(policy_action).all():
        raise ValueError("nonfinite policy action")
    unit = policy_action.clamp(-1, 1)
    raw = torch.zeros((len(unit), 7), device=unit.device, dtype=unit.dtype)
    raw[:, :3] = unit[:, :3] * cfg.translation_limit_m
    raw[:, 5] = unit[:, 3] * cfg.yaw_limit_rad
    raw[:, 6] = torch.where(unit[:, 4] > 0, 1., -1.)
    return raw, unit


def task_errors(red_pos, blue_pos, cfg=None):
    cfg = cfg or DirectPlaceConfig()
    rel = red_pos - blue_pos
    return rel[:, :2].norm(dim=-1), (rel[:, 2] - cfg.stack_height_m).abs()


def terminal_flags(red_pos, blue_pos, speed, gripper_open, stable_count,
                   bad_count, steps, cfg=None):
    """Finite-horizon task; failure remains active AFTER OPEN, independent of phase."""
    cfg = cfg or DirectPlaceConfig()
    xy, z = task_errors(red_pos, blue_pos, cfg)
    strict_now = (xy < cfg.xy_success_m) & (z < cfg.z_success_m) & gripper_open & (speed < cfg.speed_success_mps)
    stable_count = torch.where(strict_now, stable_count + 1, 0)
    bad = ((red_pos[:, 2] - blue_pos[:, 2]) < cfg.fallen_height_diff_m) | (xy > cfg.far_xy_m)
    bad_count = torch.where(bad, bad_count + 1, 0)
    success = stable_count >= cfg.stable_steps
    physical_failure = (bad_count >= cfg.failure_consecutive_steps) & ~success
    timeout = (steps >= cfg.episode_steps) & ~success & ~physical_failure
    return success, physical_failure, timeout, stable_count, bad_count, strict_now


def potential(red_pos, blue_pos, speed, gripper_open, cfg=None):
    cfg = cfg or DirectPlaceConfig()
    xy, z = task_errors(red_pos, blue_pos, cfg)
    proximity = torch.exp(-xy / .03 - z / .02)
    return (-2 * torch.tanh(xy / .03) - torch.tanh(z / .02)
            - .25 * proximity * torch.tanh(speed / .05)
            - .25 * proximity * (~gripper_open).float())


def direct_reward(prev_potential, next_potential, policy_unit, success, failed, timeout, cfg=None):
    cfg = cfg or DirectPlaceConfig()
    done = success | failed | timeout
    # Terminal Phi=0 is essential; do not accumulate proximity reward forever.
    shaping = cfg.gamma * torch.where(done, 0., next_potential) - prev_potential
    cost = cfg.step_cost + cfg.movement_cost * policy_unit[:, :4].square().sum(dim=-1)
    terminal = cfg.success_bonus * success.float() - cfg.failure_penalty * (failed | timeout).float()
    return shaping - cost + terminal, {"progress": shaping, "cost": -cost, "terminal": terminal}


def validate_training_snapshot(payload):
    if int(payload.get("source_seed", -1)) not in TRAIN_SOURCE_SEEDS:
        raise ValueError("only the five declared TRAIN source seeds may initialize M23 training")
    if payload.get("is_relative") is not True:
        raise ValueError("relative snapshot required")
    if payload.get("task") != "Isaac-Stack-Cube-Franka-IK-Rel-v0":
        raise ValueError("wrong snapshot task")
    diag = payload.get("diagnostics", {})
    if diag.get("physical_grasp") is not True or diag.get("ready") is not True:
        raise ValueError("snapshot is not a validated grasped state")
