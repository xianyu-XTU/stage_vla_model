"""Bounded development-only reference actions for the frozen 5D contract."""

from __future__ import annotations

from typing import Any, Mapping

from stage_vla_v7.interfaces import Skill

from .evaluation.success_checker import SkillTolerances


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - reference actions require torch
        raise RuntimeError("reference actions require torch") from exc
    return torch


def _vector(value: object, *, device: object | None = None) -> Any:
    torch = _torch()
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
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


def reference_action(
    skill: Skill | str,
    state: Mapping[str, object],
    *,
    tolerances: SkillTolerances = SkillTolerances(),
    translation_limit_m: float = 0.005,
    yaw_limit_rad: float = 0.02,
) -> Any:
    """Return the frozen bounded ``[dx,dy,dz,dyaw,grip]`` diagnostic action."""
    torch = _torch()
    if translation_limit_m <= 0 or yaw_limit_rad <= 0:
        raise ValueError("action limits must be positive")
    canonical = Skill(getattr(skill, "value", skill))
    measured = _normalize_state(state)
    count = measured["ee"].shape[0]
    target = measured["ee"].clone()
    grip = torch.ones(count, device=target.device)

    if canonical in {Skill.REACH, Skill.GRASP}:
        tip_midpoint = (measured["left_tip"] + measured["right_tip"]) / 2
        desired_tip_midpoint = measured["grasp_target"].clone()
        desired_tip_midpoint[:, 2] += (
            tolerances.pregrasp_tip_height_m
            if canonical is Skill.REACH
            else tolerances.grasp_contact_tip_height_m
        )
        target = measured["ee"] + (desired_tip_midpoint - tip_midpoint)
        if canonical is Skill.GRASP:
            grip = -torch.ones(count, device=target.device)
    elif canonical is Skill.LIFT:
        target[:, 2] += tolerances.lift_height_m
        grip.fill_(-1)
    elif canonical is Skill.TRANSPORT:
        target[:, :2] = measured["support"][:, :2]
        grip.fill_(-1)
    elif canonical is Skill.ALIGN:
        target[:, :2] = measured["support"][:, :2]
        target[:, 2] = measured["support"][:, 2] + measured["stack_height"] + 0.012
        grip.fill_(-1)
        relative = measured["object"] - measured["support"]
        aligned = (
            (relative[:, :2].norm(dim=-1) < tolerances.align_xy_m)
            & ((relative[:, 2] - measured["stack_height"]).abs() < 0.015)
        )
        target = torch.where(aligned[:, None], measured["ee"], target)
    elif canonical is Skill.DESCEND:
        target = measured["support"].clone()
        target[:, 2] += measured["stack_height"]
        grip.fill_(-1)
        relative = measured["object"] - measured["support"]
        descended = (
            (relative[:, :2].norm(dim=-1) < tolerances.descend_xy_m)
            & (
                (relative[:, 2] - measured["stack_height"]).abs()
                < tolerances.descend_z_m
            )
        )
        target = torch.where(descended[:, None], measured["ee"], target)
    elif canonical is Skill.RELEASE_STABILIZE:
        grip.fill_(1)
    else:
        target[:, 2] += tolerances.retreat_height_m
        grip.fill_(1)

    delta = (
        (target - measured["ee"]).clamp(-translation_limit_m, translation_limit_m)
        / translation_limit_m
    )
    action = torch.zeros((count, 5), device=target.device, dtype=target.dtype)
    action[:, :3] = delta
    action[:, 4] = grip
    return action.clamp(-1, 1)
