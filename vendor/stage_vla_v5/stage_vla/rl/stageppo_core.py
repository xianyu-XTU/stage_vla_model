"""Project StagePPO: independent PPO skills, not an external-paper reproduction.

Only terminal evidence chooses a skill. All XYZ/yaw/grip commands remain policy
outputs. Pure contracts here are testable without starting Isaac Sim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math

import torch

from .direct_place_core import (
    DirectPlaceConfig, TRAIN_SOURCE_SEEDS, potential, task_errors, terminal_flags,
)


class Skill(IntEnum):
    RELEASE_STABILIZE = 0
    RETREAT = 1


@dataclass(frozen=True)
class SkillConfig(DirectPlaceConfig):
    # Both skills keep the original stack criterion; retreat adds separation.
    retreat_distance_m: float = .10
    retreat_height_m: float = .08
    retreat_hold_steps: int = 20

    def validate(self):
        super().validate()
        if min(self.retreat_distance_m, self.retreat_height_m) <= 0:
            raise ValueError("retreat clearance must be positive")
        for name in ("stable_steps", "episode_steps", "failure_consecutive_steps", "retreat_hold_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.episode_steps < max(self.stable_steps, self.retreat_hold_steps):
            raise ValueError("horizon too short for success hold")


def interface_contract(cfg, step_dt):
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("invalid control period")
    return {
        "version": "stageppo-state46-action5-v1", "observation_dim": 46, "action_dim": 5,
        "action_order": ["dx", "dy", "dz", "dyaw", "grip"],
        "pose_frame": "robot root frame, Isaac DifferentialInverseKinematicsAction",
        "translation_limit_m": cfg.translation_limit_m, "yaw_limit_rad": cfg.yaw_limit_rad,
        "control_dt_s": step_dt, "arm_action_scale": 1., "relative_pose": True,
        "gripper": "positive OPEN; zero/negative CLOSE; actual joint position used for success",
        "roll_pitch": "zero relative increments", "observation_normalization": "fixed scales; no running normalizer",
        "previous_action_at_handoff": "preserved from upstream policy",
    }


def strict_stack(m, cfg):
    xy, z = task_errors(m["red"], m["blue"], cfg)
    return (xy < cfg.xy_success_m) & (z < cfg.z_success_m) & m["open"] & (m["speed"] < cfg.speed_success_mps)


def retreat_clear(m, cfg):
    rel = m["ee"] - m["red"]
    return (rel.norm(dim=-1) >= cfg.retreat_distance_m) & (rel[:, 2] >= cfg.retreat_height_m)


def skill_flags(skill, m, stable, bad, steps, cfg):
    skill = Skill(skill)
    if skill == Skill.RELEASE_STABILIZE:
        return terminal_flags(m["red"], m["blue"], m["speed"], m["open"], stable, bad, steps, cfg)
    strict = strict_stack(m, cfg)
    stable = torch.where(strict & retreat_clear(m, cfg), stable + 1, 0)
    xy, z = task_errors(m["red"], m["blue"], cfg)
    # Once stacked, losing stack geometry is a retreat failure, even if the
    # object has not yet fallen to the table. Grip commands are never forced.
    lost_stack = (xy >= cfg.xy_success_m) | (z >= cfg.z_success_m)
    bad = torch.where(lost_stack, bad + 1, 0)
    success = stable >= cfg.retreat_hold_steps
    failure = (bad >= cfg.failure_consecutive_steps) & ~success
    timeout = (steps >= cfg.episode_steps) & ~success & ~failure
    return success, failure, timeout, stable, bad, strict


def skill_potential(skill, m, cfg):
    value = potential(m["red"], m["blue"], m["speed"], m["open"], cfg)
    if Skill(skill) == Skill.RETREAT:
        rel = m["ee"] - m["red"]
        distance_left = (cfg.retreat_distance_m - rel.norm(dim=-1)).clamp_min(0)
        height_left = (cfg.retreat_height_m - rel[:, 2]).clamp_min(0)
        value = value - 2 * torch.tanh(distance_left / .05) - torch.tanh(height_left / .04)
    return value


def entry_valid(skill, m, cfg):
    if Skill(skill) == Skill.RETREAT:
        return strict_stack(m, cfg)
    xy, _ = task_errors(m["red"], m["blue"], cfg)
    return ((m["red"][:, 2] - m["blue"][:, 2] > cfg.fallen_height_diff_m)
            & (xy < cfg.far_xy_m) & ~m["open"])


def route_policies(policies, stage, obs):
    """No scripted fallback. Every action comes from its selected actor."""
    if stage.shape != (obs.shape[0],) or stage.dtype != torch.long:
        raise ValueError("one int64 skill id per observation required")
    if not torch.isfinite(obs).all():
        raise ValueError("nonfinite observation")
    actions = obs.new_empty((len(stage), 5))
    for value in stage.unique().tolist():
        skill = Skill(value)
        if skill not in policies:
            raise ValueError(f"missing actor for {skill.name}")
        mask = stage == value
        action = policies[skill](obs[mask])
        if action.shape != (int(mask.sum()), 5) or not torch.isfinite(action).all():
            raise ValueError("actor must return finite [N,5] actions")
        actions[mask] = action
    return actions


def chain_transition(stage, success, failure, timeout):
    for value in stage.unique().tolist():
        Skill(value)
    if torch.any(success.int() + failure.int() + timeout.int() > 1):
        raise ValueError("skill outcomes must be exclusive")
    handoff = (stage == int(Skill.RELEASE_STABILIZE)) & success
    finished_success = (stage == int(Skill.RETREAT)) & success
    next_stage = torch.where(handoff, int(Skill.RETREAT), stage)
    return next_stage, handoff, finished_success | failure | timeout


def validate_retreat_snapshot(payload, cfg, contract):
    if payload.get("version") != "stageppo-retreat-entry-v1" or payload.get("skill") != "RETREAT":
        raise ValueError("not a retreat entry snapshot")
    if payload.get("source_seed") not in TRAIN_SOURCE_SEEDS:
        raise ValueError("retreat snapshot is not from the declared TRAIN sources")
    if payload.get("task") != "Isaac-Stack-Cube-Franka-IK-Rel-v0" or payload.get("is_relative") is not True:
        raise ValueError("wrong task or non-relative retreat snapshot")
    if payload.get("interface") != contract or payload.get("task_config") != asdict(cfg):
        raise ValueError("incompatible retreat snapshot contract")
    diag = payload.get("diagnostics", {})
    if diag.get("strict_stack") is not True or diag.get("release_stable_steps", 0) < cfg.stable_steps:
        raise ValueError("retreat snapshot lacks verified upstream completion")
    if not payload.get("upstream_policy_sha256"):
        raise ValueError("missing upstream provenance")
    state = payload.get("scene_state", {})
    if "robot" not in state.get("articulation", {}) or not all(
        name in state.get("rigid_object", {}) for name in ("cube_1", "cube_2")
    ):
        raise ValueError("incomplete physical snapshot")
    action = torch.as_tensor(payload.get("previous_action", []))
    if action.shape != (5,) or not torch.isfinite(action).all() or (action.abs() > 1).any():
        raise ValueError("invalid previous action")


def summarize_evaluation(rows, count, *, chain=False):
    if count <= 0 or len(rows) != count or sorted(r["env"] for r in rows) != list(range(count)):
        raise ValueError("incomplete or duplicate evaluation denominator")
    for row in rows:
        outcomes = [row.get(k) for k in ("success", "physical_failure", "timeout")]
        if not all(isinstance(v, bool) for v in outcomes) or sum(outcomes) != 1:
            raise ValueError("exactly one explicit outcome required")
        if chain and row["success"] and not row.get("release_success"):
            raise ValueError("chain success without upstream success")
    successes = sum(r["success"] for r in rows)
    result = {"episodes": count, "successes": successes, "success_rate": successes / count,
              "physical_failures": sum(r["physical_failure"] for r in rows),
              "timeouts": sum(r["timeout"] for r in rows), "results": sorted(rows, key=lambda r: r["env"])}
    if chain:
        releases = sum(bool(r.get("release_success")) for r in rows)
        result.update(release_successes=releases, release_rate=releases / count,
                      retreat_conditional_rate=successes / releases if releases else None)
    return result
