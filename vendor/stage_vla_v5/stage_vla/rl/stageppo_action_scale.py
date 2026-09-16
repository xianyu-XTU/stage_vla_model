"""Fixed, declared per-skill actuator scaling; never a geometry controller."""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from .stageppo_core import Skill, interface_contract


def validate_retreat_limit(value):
    if not math.isfinite(value) or not .002 <= value <= .01:
        raise ValueError("retreat translation limit must be within [.002,.01] m")


def skill_interfaces(cfg, dt, retreat_limit):
    validate_retreat_limit(retreat_limit)
    return {Skill.RELEASE_STABILIZE.name: interface_contract(cfg, dt),
            Skill.RETREAT.name: interface_contract(replace(cfg, translation_limit_m=retreat_limit), dt)}


def scale_metric_actions(raw, stage, source_limit, retreat_limit):
    validate_retreat_limit(retreat_limit)
    if source_limit != .002:
        raise ValueError("adapter expects the frozen 2mm M24 decoder")
    if raw.ndim != 2 or raw.shape[1] != 7 or stage.shape != (len(raw),) or stage.dtype != torch.long:
        raise ValueError("expected [N,7] actions and [N] int64 skill ids")
    if not torch.isfinite(raw).all():
        raise ValueError("nonfinite metric action")
    for value in stage.unique().tolist():
        Skill(value)
    if (raw[:, :3].abs() > source_limit + 1e-7).any():
        raise ValueError("raw actions already exceed base decoder limit; possible double scaling")
    result = raw.clone()
    selected = stage == int(Skill.RETREAT)
    result[selected, :3] *= retreat_limit / source_limit
    # Quaternion increments and grip signal are copied exactly, for both skills.
    return result


class FixedSkillActionAdapter:
    def __init__(self, env, stage, source_limit, retreat_limit):
        validate_retreat_limit(retreat_limit)
        self.base_env, self.stage = env, stage
        self.source_limit, self.retreat_limit = source_limit, retreat_limit
        self.last_sent = None

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def step(self, raw):
        actual = scale_metric_actions(raw, self.stage, self.source_limit, self.retreat_limit)
        self.last_sent = actual.detach().clone()
        return self.base_env.step(actual)
