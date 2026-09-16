"""Bounded intermediate-goal cost for RETREAT only, not policy-invariant shaping.

An additional task objective makes partial retreat progress distinguishable even
when every sampled rollout times out. It generates rewards, never actions.
"""
from dataclasses import dataclass
import math

import torch

from .direct_place_core import task_errors


@dataclass(frozen=True)
class RetreatGoalCost:
    height: float = .06
    clearance: float = .03
    closed_gripper: float = .01
    speed: float = .01
    lost_geometry: float = .01

    def validate(self, task_cfg):
        if any(not math.isfinite(x) or x < 0 for x in vars(self).values()):
            raise ValueError("goal costs must be finite and nonnegative")
        maximum_step_cost = sum(vars(self).values()) + task_cfg.step_cost + 4 * task_cfg.movement_cost
        if task_cfg.gamma >= 1 or maximum_step_cost / (1 - task_cfg.gamma) >= task_cfg.failure_penalty:
            raise ValueError("discounted running costs could favor immediate failure")


def retreat_goal_cost(measured, cfg, weights=None):
    weights = weights or RetreatGoalCost()
    rel = measured["ee"] - measured["red"]
    height = (rel[:, 2] / cfg.retreat_height_m).clamp(0, 1)
    distance = (rel.norm(dim=-1) / cfg.retreat_distance_m).clamp(0, 1)
    # Lateral distance without vertical clearance does not earn full progress.
    clear = torch.minimum(height, distance)
    xy, z = task_errors(measured["red"], measured["blue"], cfg)
    bad_geometry = (xy >= cfg.xy_success_m) | (z >= cfg.z_success_m)
    return (weights.height * (1 - height) + weights.clearance * (1 - clear)
            + weights.closed_gripper * (~measured["open"]).float()
            + weights.speed * (measured["speed"] / cfg.speed_success_mps).clamp(0, 1)
            + weights.lost_geometry * bad_geometry.float())
