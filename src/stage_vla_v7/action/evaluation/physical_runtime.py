"""Physical-runtime terminal evaluation for the frozen cube-policy domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from stage_vla_v7.interfaces import Skill


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("physical runtime evaluation requires the 'torch' extra") from exc
    return torch


@dataclass(frozen=True)
class PhysicalSkillThresholds:
    transport_xy_m: float = 0.045
    transport_speed_mps: float = 0.05
    transport_angular_speed_radps: float = 1.0
    align_xy_m: float = 0.010
    align_height_tolerance_m: float = 0.015
    align_speed_mps: float = 0.05
    descend_height_tolerance_m: float = 0.008
    descend_speed_mps: float = 0.05
    release_xy_m: float = 0.040
    release_height_tolerance_m: float = 0.010
    release_speed_mps: float = 0.05
    retreat_distance_m: float = 0.100
    retreat_height_m: float = 0.080
    retreat_speed_mps: float = 0.05


@dataclass(frozen=True)
class PhysicalSkillEvaluation:
    ready: object
    stable_count: object
    success: object
    failure: object


def evaluate_physical_skill(
    skill: Skill | str,
    state: Mapping[str, object],
    *,
    pressure_ok: object,
    active: object,
    step_count: object,
    stable_count: object,
    stable_steps: int,
    prepress_steps: int,
    entrance_warmup_steps: int,
    entry_object_z: object,
    lift_target_height_m: object,
    align_target_height_m: object,
    descend_target_height_m: object,
    thresholds: PhysicalSkillThresholds,
) -> PhysicalSkillEvaluation:
    """Evaluate one control step with the historical physical terminal rules."""
    torch = _torch()
    canonical = Skill(skill)
    object_position = torch.as_tensor(state.get("object", state.get("red")))
    support_position = torch.as_tensor(
        state.get("support", state.get("blue")),
        device=object_position.device,
        dtype=object_position.dtype,
    )
    count = object_position.shape[0]
    if object_position.shape != (count, 3) or support_position.shape != (count, 3):
        raise ValueError("object and support positions must have shape [N,3]")

    def column(value: object, *, dtype: object | None = None):
        result = torch.as_tensor(
            value,
            device=object_position.device,
            dtype=dtype,
        ).reshape(-1)
        if result.numel() == 1:
            result = result.expand(count)
        if result.shape != (count,):
            raise ValueError(f"expected scalar or [{count}] values")
        return result

    active_t = column(active, dtype=torch.bool)
    pressure = column(pressure_ok, dtype=torch.bool)
    steps = column(step_count, dtype=torch.long)
    previous_stable = column(stable_count, dtype=torch.long)
    physical = column(state["physical"], dtype=torch.bool)
    speed = column(state["stability_speed"], dtype=object_position.dtype)
    angular_speed = column(
        state["stability_angular_speed"], dtype=object_position.dtype
    )
    relative = object_position - support_position
    xy = relative[:, :2].norm(dim=-1)
    align_target = column(align_target_height_m, dtype=object_position.dtype)
    descend_target = column(descend_target_height_m, dtype=object_position.dtype)

    if canonical is Skill.GRASP:
        ready = physical & pressure
    elif canonical is Skill.LIFT:
        entry_z = column(entry_object_z, dtype=object_position.dtype)
        lift_target = column(lift_target_height_m, dtype=object_position.dtype)
        ready = physical & pressure & (
            object_position[:, 2] - entry_z >= lift_target
        )
    elif canonical is Skill.TRANSPORT:
        ready = (
            physical
            & pressure
            & (xy <= float(thresholds.transport_xy_m))
            & (speed <= float(thresholds.transport_speed_mps))
            & (angular_speed <= float(thresholds.transport_angular_speed_radps))
        )
    elif canonical is Skill.ALIGN:
        z_error = (relative[:, 2] - align_target).abs()
        ready = (
            physical
            & pressure
            & (xy <= float(thresholds.align_xy_m))
            & (z_error <= float(thresholds.align_height_tolerance_m))
            & (speed <= float(thresholds.align_speed_mps))
        )
    elif canonical is Skill.DESCEND:
        z_error = (relative[:, 2] - descend_target).abs()
        ready = (
            physical
            & pressure
            & (xy <= float(thresholds.align_xy_m))
            & (z_error <= float(thresholds.descend_height_tolerance_m))
            & (speed <= float(thresholds.descend_speed_mps))
        )
    elif canonical is Skill.RELEASE_STABILIZE:
        z_error = (relative[:, 2] - descend_target).abs()
        ready = (
            column(state["open"], dtype=torch.bool)
            & (xy <= float(thresholds.release_xy_m))
            & (z_error <= float(thresholds.release_height_tolerance_m))
            & (speed <= float(thresholds.release_speed_mps))
        )
    elif canonical is Skill.RETREAT:
        end_effector = torch.as_tensor(
            state["ee"], device=object_position.device, dtype=object_position.dtype
        )
        ee_object = end_effector - object_position
        ready = (
            column(state["open"], dtype=torch.bool)
            & (xy <= float(thresholds.release_xy_m))
            & ((relative[:, 2] - descend_target).abs() <= float(
                thresholds.release_height_tolerance_m
            ))
            & (ee_object.norm(dim=-1) >= float(thresholds.retreat_distance_m))
            & (ee_object[:, 2] >= float(thresholds.retreat_height_m))
            & (speed <= float(thresholds.retreat_speed_mps))
        )
    else:  # pragma: no cover - Skill enum is exhaustive
        raise ValueError(f"unsupported physical Skill {canonical.value}")

    next_stable = torch.where(
        ready & active_t,
        previous_stable + 1,
        torch.zeros_like(previous_stable),
    )
    success = (next_stable >= int(stable_steps)) & active_t
    failure_guard = torch.maximum(
        torch.ones_like(steps),
        torch.full_like(steps, int(prepress_steps)),
    )
    if canonical in {Skill.ALIGN, Skill.DESCEND}:
        failure_guard = torch.maximum(
            failure_guard,
            torch.full_like(
                steps, int(prepress_steps) + int(entrance_warmup_steps)
            ),
        )
        lost = (
            ~column(state["between_fingertips"], dtype=torch.bool)
            | ~column(state["finger_a_contact"], dtype=torch.bool)
            | ~column(state["finger_b_contact"], dtype=torch.bool)
        )
        failure = lost & (steps > failure_guard) & active_t
    elif canonical in {Skill.LIFT, Skill.TRANSPORT}:
        failure = (~physical) & (steps > failure_guard) & active_t
    elif canonical in {Skill.RELEASE_STABILIZE, Skill.RETREAT}:
        failure = (
            (relative[:, 2] < 0.025) | (xy > 0.15)
        ) & active_t
    else:
        failure = torch.zeros_like(active_t)
    return PhysicalSkillEvaluation(ready, next_stable, success, failure)
