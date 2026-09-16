"""Dependency-free skill success and failure predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from stage_vla_v7.interfaces import Skill


Vector3 = tuple[float, float, float]


def _vector3(value: Vector3, name: str) -> Vector3:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _xy_distance(a: Vector3, b: Vector3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class SkillTolerances:
    grasp_distance_m: float = 0.022
    pregrasp_tip_height_m: float = 0.010
    grasp_contact_tip_height_m: float = 0.007
    pregrasp_tip_xy_m: float = 0.012
    pregrasp_tip_z_error_m: float = 0.003
    lift_height_m: float = 0.060
    transport_xy_m: float = 0.045
    align_xy_m: float = 0.010
    descend_xy_m: float = 0.012
    descend_z_m: float = 0.004
    release_xy_m: float = 0.040
    release_z_m: float = 0.010
    retreat_distance_m: float = 0.100
    retreat_height_m: float = 0.080
    speed_mps: float = 0.050
    angular_speed_radps: float = 1.0


@dataclass(frozen=True)
class SkillEvaluationState:
    end_effector_xyz_m: Vector3
    object_xyz_m: Vector3
    support_xyz_m: Vector3
    left_tip_xyz_m: Vector3 | None = None
    right_tip_xyz_m: Vector3 | None = None
    grasp_target_xyz_m: Vector3 | None = None
    object_speed_mps: float = 0.0
    object_angular_speed_radps: float = 0.0
    gripper_open: bool = True
    held: bool = False
    stack_height_m: float = 0.0468
    stable_count: int = 1

    def __post_init__(self) -> None:
        for name in ("end_effector_xyz_m", "object_xyz_m", "support_xyz_m"):
            object.__setattr__(self, name, _vector3(getattr(self, name), name))
        for name in ("left_tip_xyz_m", "right_tip_xyz_m", "grasp_target_xyz_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector3(value, name))
        scalars = (
            self.object_speed_mps,
            self.object_angular_speed_radps,
            self.stack_height_m,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("skill evaluation scalars must be finite")
        if self.stable_count < 0:
            raise ValueError("stable_count must be non-negative")


@dataclass(frozen=True)
class SuccessCheck:
    skill: Skill
    success: bool
    failure: bool
    reason: str
    metrics: Mapping[str, float | int | bool] = field(default_factory=dict)


class SuccessChecker:
    """Evaluate all eight skills using the frozen V5 physical semantics."""

    def __init__(self, tolerances: SkillTolerances | None = None) -> None:
        self.tolerances = tolerances or SkillTolerances()

    def evaluate(
        self,
        skill: Skill | str,
        state: SkillEvaluationState,
        *,
        stable_steps: int = 3,
    ) -> SuccessCheck:
        canonical = Skill(skill)
        if stable_steps < 1:
            raise ValueError("stable_steps must be positive")
        t = self.tolerances
        ee_object = _distance(state.end_effector_xyz_m, state.object_xyz_m)
        xy = _xy_distance(state.object_xyz_m, state.support_xyz_m)
        z_error = abs(
            state.object_xyz_m[2] - state.support_xyz_m[2] - state.stack_height_m
        )
        speed_ok = state.object_speed_mps < t.speed_mps
        angular_ok = state.object_angular_speed_radps < t.angular_speed_radps
        stable = state.stable_count >= stable_steps
        metrics: dict[str, float | int | bool] = {
            "ee_object_distance_m": ee_object,
            "object_support_xy_m": xy,
            "stack_z_error_m": z_error,
            "stable_count": state.stable_count,
            "held": state.held,
        }

        if canonical is Skill.REACH:
            left = state.left_tip_xyz_m or state.end_effector_xyz_m
            right = state.right_tip_xyz_m or state.end_effector_xyz_m
            target = state.grasp_target_xyz_m or state.object_xyz_m
            midpoint = tuple((a + b) / 2.0 for a, b in zip(left, right))
            tip_xy = _xy_distance(midpoint, target)  # type: ignore[arg-type]
            tip_z_error = abs(midpoint[2] - target[2] - t.pregrasp_tip_height_m)
            success = (
                tip_xy < t.pregrasp_tip_xy_m
                and tip_z_error < t.pregrasp_tip_z_error_m
                and state.gripper_open
            )
            failure = ee_object > 0.40
            metrics.update({"tip_xy_m": tip_xy, "tip_z_error_m": tip_z_error})
        elif canonical is Skill.GRASP:
            success = state.held and ee_object < t.grasp_distance_m and stable
            failure = not state.held or self._too_low(state) or xy > 0.30
        elif canonical is Skill.LIFT:
            height = state.object_xyz_m[2] - state.support_xyz_m[2]
            success = state.held and height > t.lift_height_m
            failure = not state.held or self._too_low(state) or xy > 0.30
            metrics["object_support_height_m"] = height
        elif canonical is Skill.TRANSPORT:
            success = state.held and xy < t.transport_xy_m and speed_ok and angular_ok and stable
            failure = not state.held or self._too_low(state) or xy > 0.30
        elif canonical is Skill.ALIGN:
            success = state.held and xy < t.align_xy_m and z_error < 0.015 and speed_ok and stable
            failure = not state.held or self._too_low(state) or xy > 0.30
        elif canonical is Skill.DESCEND:
            success = (
                state.held
                and xy < t.descend_xy_m
                and z_error < t.descend_z_m
                and speed_ok
                and stable
            )
            failure = not state.held or self._too_low(state) or xy > 0.30
        elif canonical is Skill.RELEASE_STABILIZE:
            success = (
                xy < t.release_xy_m
                and z_error < t.release_z_m
                and state.gripper_open
                and speed_ok
                and stable
            )
            failure = self._too_low(state) or xy > 0.15
        else:
            rel_z = state.end_effector_xyz_m[2] - state.object_xyz_m[2]
            success = ee_object >= t.retreat_distance_m and rel_z >= t.retreat_height_m
            failure = self._too_low(state) or xy > 0.15
            metrics["retreat_height_m"] = rel_z

        reason = "success" if success else "failure" if failure else "in_progress"
        return SuccessCheck(canonical, success, failure, reason, metrics)

    def evaluate_batch(
        self,
        skill: Skill | str,
        state: Mapping[str, object],
        *,
        stable_count: object = 1,
        stable_steps: int = 3,
    ) -> object:
        """Evaluate batched Torch state with the same frozen success semantics."""
        from .vectorized_success import vectorized_skill_success

        return vectorized_skill_success(
            skill,
            state,
            tolerances=self.tolerances,
            stable_count=stable_count,
            stable_steps=stable_steps,
        )

    def evaluate_failure_batch(
        self,
        skill: Skill | str,
        state: Mapping[str, object],
    ) -> object:
        """Evaluate batched Torch state with the frozen terminal-failure semantics."""
        from .vectorized_success import vectorized_skill_failure

        return vectorized_skill_failure(skill, state, tolerances=self.tolerances)

    def evaluate_physical_runtime(
        self,
        skill: Skill | str,
        state: Mapping[str, object],
        **kwargs: object,
    ) -> object:
        """Evaluate the physical VecEnv terminal contract without V5 code."""
        from .physical_runtime import evaluate_physical_skill

        return evaluate_physical_skill(skill, state, **kwargs)

    @staticmethod
    def _too_low(state: SkillEvaluationState) -> bool:
        return state.object_xyz_m[2] - state.support_xyz_m[2] < 0.025
