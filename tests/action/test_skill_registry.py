from __future__ import annotations

import math

import pytest

from stage_vla_v7.action import ACTION_REGISTRY, ActionRequest
from stage_vla_v7.action.evaluation import SkillEvaluationState, SkillEvaluator
from stage_vla_v7.interfaces import SKILL_SEQUENCE, Skill
from stage_vla_v7.orchestration import rigid_cube_profile


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_each_skill_has_complete_registry_metadata(skill: Skill) -> None:
    definition = ACTION_REGISTRY.require(skill)
    assert definition.skill is skill
    assert definition.implementation.endswith(skill.value.lower())
    assert definition.legacy_observation_dim == (52 if skill is Skill.REACH else 55)
    assert tuple(parameter.name for parameter in definition.parameter_schema) == (
        "dx",
        "dy",
        "dz",
        "dyaw",
        "grip",
    )
    assert definition.required_observation
    assert definition.success_condition
    assert definition.failure_condition


def test_unknown_skill_fails_closed() -> None:
    with pytest.raises(LookupError, match="unknown or unregistered skill"):
        ACTION_REGISTRY.require("PUSH")


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_each_skill_routes_to_a_finite_bounded_action(skill: Skill, cube_action_service) -> None:
    result = cube_action_service.act(
        ActionRequest(
            skill,
            (0.0, 0.0, 0.0),
            rigid_cube_profile("red_cube"),
            rigid_cube_profile("blue_cube"),
        )
    )
    assert len(result.action.values) == 5
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in result.action.values)
    assert result.diagnostics["skill_id"] == ACTION_REGISTRY.require(skill).skill_id


def _success_state(skill: Skill) -> SkillEvaluationState:
    base = dict(
        end_effector_xyz_m=(0.0, 0.0, 0.16),
        object_xyz_m=(0.0, 0.0, 0.0468),
        support_xyz_m=(0.0, 0.0, 0.0),
        left_tip_xyz_m=(0.0, 0.0, 0.0568),
        right_tip_xyz_m=(0.0, 0.0, 0.0568),
        gripper_open=skill in {Skill.REACH, Skill.RELEASE_STABILIZE, Skill.RETREAT},
        held=skill in {Skill.GRASP, Skill.LIFT, Skill.TRANSPORT, Skill.ALIGN, Skill.DESCEND},
        stable_count=5,
    )
    if skill is Skill.GRASP:
        base["end_effector_xyz_m"] = (0.0, 0.0, 0.0468)
    if skill is Skill.LIFT:
        base["object_xyz_m"] = (0.0, 0.0, 0.061)
    return SkillEvaluationState(**base)


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_each_skill_has_an_independent_success_predicate(skill: Skill) -> None:
    result = SkillEvaluator().evaluate(skill, _success_state(skill), stable_steps=3)
    assert result.success is True
    assert result.failure is False


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_each_skill_fails_closed_outside_its_physical_state(skill: Skill) -> None:
    if skill is Skill.REACH:
        state = SkillEvaluationState(
            end_effector_xyz_m=(1.0, 0.0, 0.0),
            object_xyz_m=(0.0, 0.0, 0.0468),
            support_xyz_m=(0.0, 0.0, 0.0),
        )
    else:
        state = SkillEvaluationState(
            end_effector_xyz_m=(0.0, 0.0, 0.0),
            object_xyz_m=(0.3, 0.0, 0.0),
            support_xyz_m=(0.0, 0.0, 0.0),
            held=False,
        )
    result = SkillEvaluator().evaluate(skill, state)
    assert result.success is False
    assert result.failure is True
