"""Registered per-skill evaluation entry point."""

from __future__ import annotations

from stage_vla_v7.interfaces import Skill

from ..action_list import ACTION_REGISTRY
from .success_checker import SkillEvaluationState, SuccessCheck, SuccessChecker


class SkillEvaluator:
    def __init__(self, checker: SuccessChecker | None = None) -> None:
        self.checker = checker or SuccessChecker()

    def evaluate(
        self,
        skill: Skill | str,
        state: SkillEvaluationState,
        *,
        stable_steps: int = 3,
    ) -> SuccessCheck:
        definition = ACTION_REGISTRY.require(skill)
        return self.checker.evaluate(definition.skill, state, stable_steps=stable_steps)


def legacy_vectorized_skill_success(skill: object, state: object, **kwargs: object) -> object:
    """Explicit compatibility adapter for the current V5 tensor evaluator."""
    from stage_vla.rl.v5_skill_contracts import skill_success

    return skill_success(skill, state, **kwargs)
