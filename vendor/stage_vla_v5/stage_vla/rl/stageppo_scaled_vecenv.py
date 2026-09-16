"""M25: retain M24 rewards/terminations, enlarge only RETREAT translation bounds."""
from __future__ import annotations

from .stageppo_vecenv import StagePPOVecEnv
from .stageppo_action_scale import FixedSkillActionAdapter, skill_interfaces, validate_retreat_limit


class ScaledStagePPOVecEnv(StagePPOVecEnv):
    def __init__(self, *args, retreat_limit_m=.01, **kwargs):
        validate_retreat_limit(retreat_limit_m)
        # Snapshot provenance and physical entry validity still checked using
        # original M24 metadata. Reuse STATE only; no actions replayed from it.
        super().__init__(*args, **kwargs)
        self.retreat_limit_m = retreat_limit_m
        self.skill_interfaces = skill_interfaces(self.task_cfg, float(self.unwrapped.step_dt), retreat_limit_m)
        self.base_interface = self.interface
        self.interface = {"version": "stageppo-per-skill-interface-v2", "skills": self.skill_interfaces}
        self.action_adapter = FixedSkillActionAdapter(self.env, self.stage, self.task_cfg.translation_limit_m, retreat_limit_m)
        self.env = self.action_adapter

    def step(self, actions):
        result = super().step(actions)
        # Parent stores pre-adapter actions. Expose actual executed metric values.
        self.last_raw_action.copy_(self.action_adapter.last_sent)
        return result

    def statistics(self):
        result = super().statistics()
        result["action_interfaces"] = self.skill_interfaces
        return result
