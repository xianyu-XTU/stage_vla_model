"""M27 release-only training objective; no action or success gate."""
from dataclasses import dataclass
import math

import torch

from .direct_place_core import task_errors


@dataclass(frozen=True)
class ReleaseQuality:
    xy_scale_m: float = .015
    angular_scale_radps: float = 1.
    terminal_discount_fraction: float = .8
    xy_running_cost: float = .03
    speed_running_cost: float = .01
    angular_running_cost: float = .01

    def validate(self, cfg):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("nonfinite release quality weight")
        if min(self.xy_scale_m, self.angular_scale_radps) <= 0 or not 0 <= self.terminal_discount_fraction < 1:
            raise ValueError("positive scales and success-bonus fraction below one required")
        costs = [self.xy_running_cost, self.speed_running_cost, self.angular_running_cost]
        if min(costs) < 0 or not 0 < cfg.gamma < 1:
            raise ValueError("invalid running costs or discount")
        if (sum(costs) + cfg.step_cost + 4 * cfg.movement_cost) / (1 - cfg.gamma) >= cfg.failure_penalty:
            raise ValueError("running-cost magnitude could favor immediate failure")


def release_quality(m, cfg, weights=None):
    w = weights or ReleaseQuality()
    xy, _ = task_errors(m["red"], m["blue"], cfg)
    angular = m["angular"].norm(dim=-1)
    # Smooth credit for a centered, low-motion exit. This is a changed reward
    # objective, NOT policy-invariant shaping or a different success definition.
    quality = torch.exp(-(xy / w.xy_scale_m).square()
                        - (m["speed"] / cfg.speed_success_mps).square()
                        - (angular / w.angular_scale_radps).square())
    return quality


def release_quality_cost(m, success, cfg, weights=None):
    w = weights or ReleaseQuality()
    xy, _ = task_errors(m["red"], m["blue"], cfg)
    running = (w.xy_running_cost * (xy / w.xy_scale_m).clamp(0, 1)
               + w.speed_running_cost * (m["speed"] / cfg.speed_success_mps).clamp(0, 1)
               + w.angular_running_cost * (m["angular"].norm(dim=-1) / w.angular_scale_radps).clamp(0, 1))
    terminal = cfg.success_bonus * w.terminal_discount_fraction * (1 - release_quality(m, cfg, w)) * success
    return running + terminal
