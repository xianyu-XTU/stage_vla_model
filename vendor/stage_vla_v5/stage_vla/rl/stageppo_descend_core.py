"""Independent StagePPO DESCEND task contract."""
from dataclasses import asdict, dataclass

import torch

from .direct_place_core import DirectPlaceConfig, direct_reward, validate_training_snapshot
from .stageppo_align_v2_core import GRASP_V2_CONFIG, align_v2_grasp
from .stageppo_core import interface_contract


@dataclass(frozen=True)
class DescendConfig(DirectPlaceConfig):
    translation_limit_m: float = 0.003
    stack_height_m: float = 0.0468
    xy_success_m: float = 0.012
    z_success_m: float = 0.004
    speed_success_mps: float = 0.05
    stable_steps: int = 5
    episode_steps: int = 180
    fallen_height_diff_m: float = 0.034
    far_xy_m: float = 0.05
    max_angular_speed_radps: float = 1.0
    grasp_grace_steps: int = 2
    xy_cost: float = 0.04
    z_cost: float = 0.04
    grasp_cost: float = 0.03
    motion_cost: float = 0.01

    def validate(self):
        super().validate()
        if min(self.xy_success_m, self.z_success_m, self.max_angular_speed_radps) <= 0:
            raise ValueError("DESCEND tolerances must be positive")
        if self.far_xy_m <= self.xy_success_m:
            raise ValueError("DESCEND far threshold must exceed success tolerance")
        if not isinstance(self.grasp_grace_steps, int) or self.grasp_grace_steps < 1:
            raise ValueError("DESCEND grasp grace must be a positive integer")
        costs = self.xy_cost + self.z_cost + self.grasp_cost + self.motion_cost
        if min(self.xy_cost, self.z_cost, self.grasp_cost, self.motion_cost) < 0:
            raise ValueError("DESCEND costs must be nonnegative")
        if (costs + self.step_cost + 4 * self.movement_cost) / (1 - self.gamma) >= self.failure_penalty:
            raise ValueError("DESCEND running-cost bound exceeds immediate failure penalty")


def descend_errors(measured, cfg):
    relative = measured["red"] - measured["blue"]
    return relative[:, :2].norm(dim=-1), relative[:, 2], (relative[:, 2] - cfg.stack_height_m).abs()


def descend_flags(measured, stable, bad, steps, cfg):
    xy, height, z = descend_errors(measured, cfg)
    held, _, _ = align_v2_grasp(measured)
    angular = measured["angular"].norm(dim=-1)
    strict = (held & (xy < cfg.xy_success_m) & (z < cfg.z_success_m)
              & (measured["speed"] < cfg.speed_success_mps)
              & (angular < cfg.max_angular_speed_radps))
    stable = torch.where(strict, stable + 1, 0)
    lost = (~held) & (steps > cfg.grasp_grace_steps)
    too_low = height < cfg.fallen_height_diff_m
    far = (xy > cfg.far_xy_m) | (height > 0.10)
    bad = torch.where(lost | too_low | far, bad + 1, 0)
    success = stable >= cfg.stable_steps
    failure = (bad >= cfg.failure_consecutive_steps) & ~success
    timeout = (steps >= cfg.episode_steps) & ~success & ~failure
    diagnostics = {
        "held": held, "strict": strict, "lost_grasp": lost,
        "too_low": too_low, "far": far, "actual_open": measured["open"],
        "xy": xy, "height": height, "z_error": z,
    }
    return success, failure, timeout, stable, bad, diagnostics


def descend_potential(measured, cfg):
    xy, _, z = descend_errors(measured, cfg)
    held, _, _ = align_v2_grasp(measured)
    return -2 * torch.tanh(xy / 0.02) - 2 * torch.tanh(z / 0.012) - 0.5 * (~held).float()


def descend_reward(before, after, unit, success, failure, timeout, cfg):
    reward, _ = direct_reward(descend_potential(before, cfg), descend_potential(after, cfg),
                              unit, success, failure, timeout, cfg)
    xy, _, z = descend_errors(after, cfg)
    held, _, _ = align_v2_grasp(after)
    cost = (cfg.xy_cost * (xy / 0.02).clamp(0, 1)
            + cfg.z_cost * (z / 0.012).clamp(0, 1)
            + cfg.grasp_cost * (~held).float()
            + cfg.motion_cost * (after["speed"] / cfg.speed_success_mps).clamp(0, 1))
    return reward - cost


def descend_interface(cfg, dt):
    base = interface_contract(cfg, dt)
    return {
        **base, "version": "stageppo-descend-control-speed-state52-action5-v1",
        "skill": "DESCEND", "observation_dim": 52,
        "observation_semantics": "state52; first XYZ is red-blue stack-goal error / .05",
        "success_linear_speed": "control-step red displacement / control_dt",
        "roll_pitch": "fixed reset-time reference; actor owns XYZ/yaw/grip",
        "grasp_config": asdict(GRASP_V2_CONFIG),
    }


def validate_descend_snapshot(payload):
    validate_training_snapshot(payload)
    if payload.get("snapshot_role") != "descend" or payload.get("initial_phase") != 1:
        raise ValueError("DESCEND requires a phase-1 descend snapshot")
    if payload.get("diagnostics", {}).get("align_v3_success") is not True:
        raise ValueError("DESCEND entry must come from a verified ALIGN-v3 success")


def descend_teacher_action(measured, cfg, deadband_xy_m=0.004, deadband_z_m=0.002):
    """Geometric demonstration action; never used by the deployed policy."""
    if not 0 < deadband_xy_m < cfg.xy_success_m or not 0 < deadband_z_m < cfg.z_success_m:
        raise ValueError("teacher deadbands must lie inside success tolerances")
    target = measured["blue"] + measured["blue"].new_tensor([0.0, 0.0, cfg.stack_height_m])
    error = target - measured["red"]
    metric = error.clamp(-cfg.translation_limit_m, cfg.translation_limit_m)
    settled = (error[:, :2].norm(dim=-1) < deadband_xy_m) & (error[:, 2].abs() < deadband_z_m)
    metric[settled] = 0.0
    action = measured["red"].new_zeros((len(error), 5))
    action[:, :3] = metric / cfg.translation_limit_m
    action[:, 4] = -1.0
    return action, settled
