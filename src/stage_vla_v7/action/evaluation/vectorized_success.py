"""Native batched Torch predicates for the frozen physical Skill contract."""

from __future__ import annotations

from typing import Any, Mapping

from stage_vla_v7.interfaces import Skill

from .success_checker import SkillTolerances


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("vectorized success evaluation requires the 'torch' extra") from exc
    return torch


def _vector(
    value: object,
    *,
    device: object | None = None,
    dtype: object | None = None,
) -> Any:
    torch = _torch()
    tensor = torch.as_tensor(
        value,
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != 3:
        raise ValueError("positions must have shape [N,3]")
    return tensor


def _column(
    value: object,
    count: int,
    *,
    device: object | None = None,
    dtype: object | None = None,
) -> Any:
    torch = _torch()
    tensor = torch.as_tensor(
        value,
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    ).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(count)
    if tensor.shape != (count,):
        raise ValueError(f"expected scalar or [{count}] values")
    return tensor


def _normalize_state(state: Mapping[str, object]) -> dict[str, Any]:
    torch = _torch()
    if "ee" not in state:
        raise ValueError("missing state fields: ['ee']")
    ee = _vector(state["ee"])
    object_key = "object" if "object" in state else "red"
    support_key = "support" if "support" in state else "blue"
    if object_key not in state or support_key not in state:
        raise ValueError("missing state fields: object/support (legacy red/blue aliases accepted)")

    manipulated = _vector(state[object_key], device=ee.device)
    support = _vector(state[support_key], device=ee.device)
    if not (ee.shape == manipulated.shape == support.shape):
        raise ValueError("ee, object and support must have matching shapes")

    count = ee.shape[0]
    left_tip = _vector(state.get("left_tip", ee), device=ee.device)
    right_tip = _vector(state.get("right_tip", ee), device=ee.device)
    if not (left_tip.shape == right_tip.shape == ee.shape):
        raise ValueError("finger-tip positions must match ee shape")

    grasp_target = _vector(state.get("grasp_target", manipulated), device=ee.device)
    if grasp_target.shape != ee.shape:
        raise ValueError("grasp_target must match ee shape")

    return {
        "ee": ee,
        "object": manipulated,
        "support": support,
        "left_tip": left_tip,
        "right_tip": right_tip,
        "grasp_target": grasp_target,
        "speed": _column(state.get("speed", 0.0), count, device=ee.device),
        "angular_speed": _column(
            state.get("angular_speed", 0.0), count, device=ee.device
        ),
        "open": _column(
            state.get("open", True), count, device=ee.device, dtype=torch.bool
        ),
        "held": _column(
            state.get("held", False), count, device=ee.device, dtype=torch.bool
        ),
        "stack_height": _column(
            state.get("stack_height", 0.0468), count, device=ee.device
        ),
    }


def vectorized_skill_success(
    skill: Skill | str,
    state: Mapping[str, object],
    *,
    tolerances: SkillTolerances = SkillTolerances(),
    stable_count: object = 1,
    stable_steps: int = 3,
) -> Any:
    """Evaluate the V5-compatible success predicate without importing V5 code."""
    torch = _torch()
    canonical = Skill(skill)
    measured = _normalize_state(state)
    count = measured["ee"].shape[0]
    stable = _column(
        stable_count,
        count,
        device=measured["ee"].device,
        dtype=torch.long,
    )
    ee_object = (measured["ee"] - measured["object"]).norm(dim=-1)
    object_support = measured["object"] - measured["support"]
    xy = object_support[:, :2].norm(dim=-1)
    z_error = (object_support[:, 2] - measured["stack_height"]).abs()
    speed_ok = measured["speed"] < tolerances.speed_mps
    angular_speed_ok = measured["angular_speed"] < tolerances.angular_speed_radps

    if canonical is Skill.REACH:
        tip_midpoint = (measured["left_tip"] + measured["right_tip"]) / 2
        tip_error = tip_midpoint - measured["grasp_target"]
        return (
            (tip_error[:, :2].norm(dim=-1) < tolerances.pregrasp_tip_xy_m)
            & (
                (tip_error[:, 2] - tolerances.pregrasp_tip_height_m).abs()
                < tolerances.pregrasp_tip_z_error_m
            )
            & measured["open"]
        )
    if canonical is Skill.GRASP:
        return (
            measured["held"]
            & (ee_object < tolerances.grasp_distance_m)
            & (stable >= stable_steps)
        )
    if canonical is Skill.LIFT:
        return measured["held"] & (
            object_support[:, 2] > tolerances.lift_height_m
        )
    if canonical is Skill.TRANSPORT:
        return (
            measured["held"]
            & (xy < tolerances.transport_xy_m)
            & speed_ok
            & angular_speed_ok
            & (stable >= stable_steps)
        )
    if canonical is Skill.ALIGN:
        return (
            measured["held"]
            & (xy < tolerances.align_xy_m)
            & (z_error < 0.015)
            & speed_ok
            & (stable >= stable_steps)
        )
    if canonical is Skill.DESCEND:
        return (
            measured["held"]
            & (xy < tolerances.descend_xy_m)
            & (z_error < tolerances.descend_z_m)
            & speed_ok
            & (stable >= stable_steps)
        )
    if canonical is Skill.RELEASE_STABILIZE:
        return (
            (xy < tolerances.release_xy_m)
            & (z_error < tolerances.release_z_m)
            & measured["open"]
            & speed_ok
            & (stable >= stable_steps)
        )
    relative = measured["ee"] - measured["object"]
    return (
        (relative.norm(dim=-1) >= tolerances.retreat_distance_m)
        & (relative[:, 2] >= tolerances.retreat_height_m)
    )


def vectorized_skill_failure(
    skill: Skill | str,
    state: Mapping[str, object],
    *,
    tolerances: SkillTolerances = SkillTolerances(),
) -> Any:
    """Evaluate the V5-compatible terminal-failure predicate without V5 code."""
    canonical = Skill(skill)
    measured = _normalize_state(state)
    ee_object = (measured["ee"] - measured["object"]).norm(dim=-1)
    object_support = measured["object"] - measured["support"]
    xy = object_support[:, :2].norm(dim=-1)
    too_low = object_support[:, 2] < 0.025

    if canonical in {
        Skill.GRASP,
        Skill.LIFT,
        Skill.TRANSPORT,
        Skill.ALIGN,
        Skill.DESCEND,
    }:
        return (~measured["held"]) | too_low | (xy > 0.30)
    if canonical is Skill.REACH:
        return ee_object > 0.40
    return too_low | (xy > 0.15)
