from __future__ import annotations

import pytest

from stage_vla_v7.action import ActionBundle, ActionRequest, ActionRouter
from stage_vla_v7.contracts import ModelDescriptor, RobotAction, RoutingError, Skill
from stage_vla_v7.orchestration import rigid_cube_profile


def test_action_service_routes_and_projects(cube_action_service) -> None:
    red = rigid_cube_profile("red_cube")
    blue = rigid_cube_profile("blue_cube")
    result = cube_action_service.act(
        ActionRequest(Skill.GRASP, (0.0, 0.0, 0.0), red, blue)
    )
    assert result.action.values == (1.0, -1.0, 0.25, 1.0, -0.5)
    assert result.diagnostics["safety_projected"] is True


def test_finished_action_has_zero_motion(cube_action_service) -> None:
    red = rigid_cube_profile("red_cube")
    blue = rigid_cube_profile("blue_cube")
    result = cube_action_service.act(
        ActionRequest(Skill.RETREAT, (0.0, 0.0, 0.0), red, blue, finished=True)
    )
    assert result.action.values == (0.0, 0.0, 0.0, 0.0, 1.0)


def test_router_rejects_ambiguous_domains(cube_action_service) -> None:
    original = cube_action_service.router.bundles["cube"]
    duplicate = ActionBundle(
        ModelDescriptor("duplicate", "1", "action-bundle"),
        original.domain,
        original.policies,
    )
    router = ActionRouter({"one": original, "two": duplicate})
    with pytest.raises(RoutingError, match="ambiguous"):
        router.route(rigid_cube_profile("red_cube"), rigid_cube_profile("blue_cube"))


def test_policy_observation_dimension_is_enforced(cube_action_service) -> None:
    with pytest.raises(ValueError, match="observation"):
        cube_action_service.act(
            ActionRequest(
                Skill.REACH,
                (0.0,),
                rigid_cube_profile("red_cube"),
                rigid_cube_profile("blue_cube"),
            )
        )
