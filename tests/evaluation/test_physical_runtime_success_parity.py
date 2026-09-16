from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from stage_vla_v7.action.evaluation import (
    PhysicalSkillThresholds,
    SuccessChecker,
)
from stage_vla_v7.interfaces import SKILL_SEQUENCE, Skill


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.skill_action_safety import carrying_grasp_lost  # noqa: E402
from stage_vla.rl.transport_handoff import (  # noqa: E402
    TransportHandoffConfig,
    transport_handoff_ready,
)


def _inputs(seed: int, count: int = 97):
    generator = torch.Generator().manual_seed(seed)
    obj = torch.randn(count, 3, generator=generator) * 0.1
    support = torch.randn(count, 3, generator=generator) * 0.1
    state = {
        "red": obj,
        "blue": support,
        "ee": obj + torch.randn(count, 3, generator=generator) * 0.1,
        "open": torch.rand(count, generator=generator) > 0.5,
        "physical": torch.rand(count, generator=generator) > 0.35,
        "between_fingertips": torch.rand(count, generator=generator) > 0.2,
        "finger_a_contact": torch.rand(count, generator=generator) > 0.2,
        "finger_b_contact": torch.rand(count, generator=generator) > 0.2,
        "stability_speed": torch.rand(count, generator=generator) * 0.1,
        "stability_angular_speed": torch.rand(count, generator=generator) * 2.0,
    }
    values = {
        "pressure_ok": torch.rand(count, generator=generator) > 0.25,
        "active": torch.rand(count, generator=generator) > 0.15,
        "step_count": torch.randint(0, 12, (count,), generator=generator),
        "stable_count": torch.randint(0, 5, (count,), generator=generator),
        "entry_object_z": torch.randn(count, generator=generator) * 0.05,
        "lift_target_height_m": torch.rand(count, generator=generator) * 0.08,
        "align_target_height_m": torch.rand(count, generator=generator) * 0.08,
        "descend_target_height_m": torch.rand(count, generator=generator) * 0.08,
    }
    return state, values


def _legacy(skill: Skill, state, values, thresholds):
    active = values["active"]
    pressure_ok = values["pressure_ok"]
    relative = state["red"] - state["blue"]
    xy = relative[:, :2].norm(dim=-1)
    if skill is Skill.GRASP:
        ready = state["physical"] & pressure_ok
    elif skill is Skill.LIFT:
        ready = (
            state["physical"]
            & pressure_ok
            & (
                state["red"][:, 2] - values["entry_object_z"]
                >= values["lift_target_height_m"]
            )
        )
    elif skill is Skill.TRANSPORT:
        ready = transport_handoff_ready(
            state["physical"],
            pressure_ok,
            xy,
            state["stability_speed"],
            state["stability_angular_speed"],
            cfg=TransportHandoffConfig(
                success_xy_m=thresholds.transport_xy_m,
                linear_speed_mps=thresholds.transport_speed_mps,
                angular_speed_radps=thresholds.transport_angular_speed_radps,
                approach_width_m=thresholds.transport_xy_m,
            ),
        )
    elif skill is Skill.ALIGN:
        ready = (
            state["physical"]
            & pressure_ok
            & (xy <= thresholds.align_xy_m)
            & (
                (
                    relative[:, 2] - values["align_target_height_m"]
                ).abs()
                <= thresholds.align_height_tolerance_m
            )
            & (state["stability_speed"] <= thresholds.align_speed_mps)
        )
    elif skill is Skill.DESCEND:
        ready = (
            state["physical"]
            & pressure_ok
            & (xy <= thresholds.align_xy_m)
            & (
                (
                    relative[:, 2] - values["descend_target_height_m"]
                ).abs()
                <= thresholds.descend_height_tolerance_m
            )
            & (state["stability_speed"] <= thresholds.descend_speed_mps)
        )
    elif skill is Skill.RELEASE_STABILIZE:
        ready = (
            state["open"]
            & (xy <= thresholds.release_xy_m)
            & (
                (
                    relative[:, 2] - values["descend_target_height_m"]
                ).abs()
                <= thresholds.release_height_tolerance_m
            )
            & (state["stability_speed"] <= thresholds.release_speed_mps)
        )
    else:
        ee_object = state["ee"] - state["red"]
        ready = (
            state["open"]
            & (xy <= thresholds.release_xy_m)
            & (
                (
                    relative[:, 2] - values["descend_target_height_m"]
                ).abs()
                <= thresholds.release_height_tolerance_m
            )
            & (ee_object.norm(dim=-1) >= thresholds.retreat_distance_m)
            & (ee_object[:, 2] >= thresholds.retreat_height_m)
            & (state["stability_speed"] <= thresholds.retreat_speed_mps)
        )
    stable = torch.where(
        ready & active,
        values["stable_count"] + 1,
        torch.zeros_like(values["stable_count"]),
    )
    success = (stable >= 3) & active
    guard = torch.maximum(
        torch.ones_like(values["step_count"]),
        torch.full_like(values["step_count"], 2),
    )
    if skill in {Skill.ALIGN, Skill.DESCEND}:
        guard = torch.maximum(guard, torch.full_like(guard, 6))
        failure = carrying_grasp_lost(
            state["between_fingertips"],
            state["finger_a_contact"],
            state["finger_b_contact"],
        ) & (values["step_count"] > guard) & active
    elif skill in {Skill.LIFT, Skill.TRANSPORT}:
        failure = (
            ~state["physical"]
            & (values["step_count"] > guard)
            & active
        )
    elif skill in {Skill.RELEASE_STABILIZE, Skill.RETREAT}:
        failure = (
            (relative[:, 2] < 0.025) | (xy > 0.15)
        ) & active
    else:
        failure = torch.zeros_like(active)
    return ready, stable, success, failure


@pytest.mark.parametrize("seed", [1, 29, 61081])
@pytest.mark.parametrize("skill", SKILL_SEQUENCE[1:])
def test_physical_runtime_terminal_matches_v5(skill: Skill, seed: int) -> None:
    state, values = _inputs(seed)
    thresholds = PhysicalSkillThresholds()
    expected = _legacy(skill, state, values, thresholds)
    actual = SuccessChecker().evaluate_physical_runtime(
        skill,
        state,
        pressure_ok=values["pressure_ok"],
        active=values["active"],
        step_count=values["step_count"],
        stable_count=values["stable_count"],
        stable_steps=3,
        prepress_steps=2,
        entrance_warmup_steps=4,
        entry_object_z=values["entry_object_z"],
        lift_target_height_m=values["lift_target_height_m"],
        align_target_height_m=values["align_target_height_m"],
        descend_target_height_m=values["descend_target_height_m"],
        thresholds=thresholds,
    )
    for observed, reference in zip(
        (actual.ready, actual.stable_count, actual.success, actual.failure),
        expected,
    ):
        assert observed.shape == reference.shape
        assert observed.dtype == reference.dtype
        assert torch.equal(observed, reference)
