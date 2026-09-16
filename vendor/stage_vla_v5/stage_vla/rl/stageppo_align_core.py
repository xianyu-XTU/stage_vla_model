"""Independent ALIGN contracts. No controller or Isaac imports."""
from dataclasses import asdict, dataclass

import torch

from .direct_place_core import DirectPlaceConfig, direct_reward, validate_training_snapshot
from .stageppo_core import interface_contract
from stage_vla.stages.physical_grasp import PhysicalGraspConfig, physical_grasp_diagnostics


@dataclass(frozen=True)
class AlignConfig(DirectPlaceConfig):
    translation_limit_m: float = .005
    xy_success_m: float = .010
    stable_steps: int = 5
    episode_steps: int = 180
    # Height refers to red center - blue center, not a measured mesh clearance.
    height_target_m: float = .0618
    height_low_m: float = .0538
    height_high_m: float = .0768
    height_failure_m: float = .0498
    max_angular_speed_radps: float = 1.
    grasp_grace_steps: int = 2
    xy_cost: float = .04
    height_cost: float = .02
    grasp_cost: float = .03
    motion_cost: float = .01

    def validate(self):
        super().validate()
        if not self.height_failure_m < self.height_low_m < self.height_target_m < self.height_high_m:
            raise ValueError("invalid ALIGN height bands")
        if min(self.xy_success_m, self.max_angular_speed_radps) <= 0:
            raise ValueError("positive ALIGN tolerances required")
        for name in ("stable_steps", "episode_steps", "failure_consecutive_steps", "grasp_grace_steps"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError("positive integer counters required")
        costs = [self.xy_cost, self.height_cost, self.grasp_cost, self.motion_cost]
        if min(costs) < 0 or not 0 < self.gamma < 1 or self.episode_steps < self.stable_steps:
            raise ValueError("invalid ALIGN horizon or cost")
        if (sum(costs) + self.step_cost + 4 * self.movement_cost) / (1 - self.gamma) >= self.failure_penalty:
            raise ValueError("running-cost bound exceeds immediate failure penalty")


GRASP_CONFIG = PhysicalGraspConfig(radial_tolerance_m=.03, height_tolerance_m=.012,
                                  contact_force_threshold_n=.5, endpoint_margin=.05)


def align_grasp(m):
    d = physical_grasp_diagnostics(m["red"], m["left_tip"], m["right_tip"],
        m["force"][:, 0], m["force"][:, 1], cfg=GRASP_CONFIG)
    geometry = d.between_fingertips & d.left_height_aligned & d.right_height_aligned
    # Net force is not contact-pair identity. Combine the existing geometric
    # predicate with measured width; validate positive/negative probes first.
    # A rotated block can legitimately span >65mm. Require BOTH measured
    # fingers below the 40mm open target minus its 2mm tolerance instead.
    narrow = (m["grip"] < .038).all(dim=-1) & ~m["open"]
    return d.physical_grasp & narrow, geometry & narrow, d


def align_errors(m, c):
    rel = m["red"] - m["blue"]
    return rel[:, :2].norm(dim=-1), rel[:, 2], (rel[:, 2] - c.height_target_m).abs()


def align_flags(m, stable, bad, steps, c):
    xy, height, _ = align_errors(m, c)
    held, _, _ = align_grasp(m)
    angular = m["angular"].norm(dim=-1)
    strict = (held & (xy < c.xy_success_m) & (height >= c.height_low_m) & (height <= c.height_high_m)
              & (m["speed"] < c.speed_success_mps) & (angular < c.max_angular_speed_radps))
    stable = torch.where(strict, stable + 1, 0)
    lost = (~held) & (steps > c.grasp_grace_steps)
    dropped = height < c.height_failure_m
    far = (xy > c.far_xy_m) | (height > .15)
    bad = torch.where(lost | dropped | far, bad + 1, 0)
    success = stable >= c.stable_steps
    failure = (bad >= c.failure_consecutive_steps) & ~success
    timeout = (steps >= c.episode_steps) & ~success & ~failure
    return success, failure, timeout, stable, bad, {
        "held": held, "strict": strict, "lost_grasp": lost, "low_height": dropped,
        "far": far, "actual_open": m["open"], "xy": xy, "height": height}


def align_potential(m, c):
    xy, _, z = align_errors(m, c)
    held, _, _ = align_grasp(m)
    return -2 * torch.tanh(xy / .03) - torch.tanh(z / .02) - .5 * (~held).float()


def align_reward(before, after, unit, success, failure, timeout, c):
    reward, _ = direct_reward(align_potential(before, c), align_potential(after, c), unit, success, failure, timeout, c)
    xy, _, z = align_errors(after, c)
    held, _, _ = align_grasp(after)
    cost = (c.xy_cost * (xy / .03).clamp(0, 1) + c.height_cost * (z / .02).clamp(0, 1)
            + c.grasp_cost * (~held).float() + c.motion_cost * (after["speed"] / .05).clamp(0, 1))
    return reward - cost


def align_interface(c, dt):
    base = interface_contract(c, dt)
    return {**base, "version": "stageppo-align-state52-action5-v1", "skill": "ALIGN", "observation_dim": 52,
        "observation_semantics": "M23 scales; Z error targets ALIGN hover height; append left/right fingertip minus red / .05",
        "snapshot_previous_action": "zero bookkeeping; no source action replay", "grasp_config": asdict(GRASP_CONFIG),
        "grasp_finger_max_m": .038}


def validate_align_snapshot(p):
    validate_training_snapshot(p)
    if p.get("snapshot_role") != "align" or p.get("initial_phase") != 0:
        raise ValueError("ALIGN must use original TRAIN align entries, not release/retreat states")
