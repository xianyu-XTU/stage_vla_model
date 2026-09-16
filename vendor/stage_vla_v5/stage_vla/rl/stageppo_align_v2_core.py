"""Versioned ALIGN-v2 task logic after M30/M31 grasp and tilt calibration."""
from dataclasses import asdict

import torch

from .direct_place_core import direct_reward
from .grip_calibration import CONTRACT as GRASP_V2_CONFIG, shadow_grasp
from .stageppo_align_core import AlignConfig, align_errors
from .stageppo_core import interface_contract


def align_v2_grasp(m):
    old, held, diagnostics = shadow_grasp(m, GRASP_V2_CONFIG)
    return held, old, diagnostics


def align_v2_entry_geometry(m):
    """Reset-time geometry without stale post-restore contact sensor values."""
    _, _, diagnostics = shadow_grasp(m, GRASP_V2_CONFIG)
    plausible = ((m["grip"] >= GRASP_V2_CONFIG.joint_min_m)
                 & (m["grip"] <= GRASP_V2_CONFIG.joint_max_m)).all(-1)
    closure = (m["grip"] < GRASP_V2_CONFIG.one_finger_below_m).any(-1)
    geometry = (diagnostics.between_fingertips & diagnostics.left_height_aligned
                & diagnostics.right_height_aligned)
    return geometry & plausible & closure & ~m["open"]


def align_v2_flags(m, stable, bad, steps, cfg):
    xy, height, _ = align_errors(m, cfg)
    held, _, _ = align_v2_grasp(m)
    angular = m["angular"].norm(dim=-1)
    strict = (held & (xy < cfg.xy_success_m) & (height >= cfg.height_low_m)
              & (height <= cfg.height_high_m) & (m["speed"] < cfg.speed_success_mps)
              & (angular < cfg.max_angular_speed_radps))
    stable = torch.where(strict, stable + 1, 0)
    lost = (~held) & (steps > cfg.grasp_grace_steps)
    dropped = height < cfg.height_failure_m
    far = (xy > cfg.far_xy_m) | (height > .15)
    bad = torch.where(lost | dropped | far, bad + 1, 0)
    success = stable >= cfg.stable_steps
    failure = (bad >= cfg.failure_consecutive_steps) & ~success
    timeout = (steps >= cfg.episode_steps) & ~success & ~failure
    return success, failure, timeout, stable, bad, {"held": held, "strict": strict,
        "lost_grasp": lost, "low_height": dropped, "far": far,
        "actual_open": m["open"], "xy": xy, "height": height}


def align_v2_potential(m, cfg):
    xy, _, z = align_errors(m, cfg)
    held, _, _ = align_v2_grasp(m)
    return -2 * torch.tanh(xy / .03) - torch.tanh(z / .02) - .5 * (~held).float()


def align_v2_reward(before, after, unit, success, failure, timeout, cfg):
    reward, _ = direct_reward(align_v2_potential(before, cfg), align_v2_potential(after, cfg),
                              unit, success, failure, timeout, cfg)
    xy, _, z = align_errors(after, cfg)
    held, _, _ = align_v2_grasp(after)
    cost = (cfg.xy_cost * (xy / .03).clamp(0, 1) + cfg.height_cost * (z / .02).clamp(0, 1)
            + cfg.grasp_cost * (~held).float()
            + cfg.motion_cost * (after["speed"] / .05).clamp(0, 1))
    return reward - cost


def align_v2_interface(cfg, dt):
    base = interface_contract(cfg, dt)
    return {**base, "version": "stageppo-align-v2-fixed-tilt-state52-action5-v1",
        "skill": "ALIGN", "observation_dim": 52,
        "observation_semantics": "M28 state52; first XYZ is red-blue hover-goal error / .05",
        "roll_pitch": "reset-time reference held by project-local IK; actor still owns XYZ/yaw/grip",
        "grasp_config": asdict(GRASP_V2_CONFIG),
        "grasp_change": "M30 calibrated asymmetric measured closure; old result logged separately"}
