"""Full-task stable-stack acceptance built from public skill predicates."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import Skill

from .success_checker import SkillEvaluationState, SuccessChecker


@dataclass(frozen=True)
class TaskEvaluation:
    success: bool
    stable_stack: bool
    retreat_clear: bool
    reason: str


class TaskEvaluator:
    def __init__(self, checker: SuccessChecker | None = None) -> None:
        self.checker = checker or SuccessChecker()

    def evaluate(self, state: SkillEvaluationState, *, stable_steps: int = 5) -> TaskEvaluation:
        release = self.checker.evaluate(
            Skill.RELEASE_STABILIZE,
            state,
            stable_steps=stable_steps,
        )
        retreat = self.checker.evaluate(Skill.RETREAT, state, stable_steps=stable_steps)
        success = release.success and retreat.success
        return TaskEvaluation(
            success,
            release.success,
            retreat.success,
            "success" if success else "stack_or_retreat_not_stable",
        )
