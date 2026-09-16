"""Shared dependency-free test fixtures."""

from __future__ import annotations

import pytest

from stage_vla_v7.action import (
    ActionBundle,
    ActionRouter,
    ActionService,
    ConstantActionPolicy,
    PolicyDomain,
)
from stage_vla_v7.contracts import ModelDescriptor, RobotAction, SKILL_SEQUENCE


@pytest.fixture
def cube_action_service() -> ActionService:
    policies = {
        skill: ConstantActionPolicy(
            RobotAction(2.0, -2.0, 0.25, 1.5, -0.5),
            observation_dim=3,
            name=f"test-{skill.value.lower()}",
        )
        for skill in SKILL_SEQUENCE
    }
    domain = PolicyDomain(
        object_geometries=("box",),
        support_geometries=("box",),
        object_size_min_m=(0.04, 0.04, 0.04),
        object_size_max_m=(0.04, 0.04, 0.04),
        support_size_min_m=(0.04, 0.04, 0.04),
        support_size_max_m=(0.04, 0.04, 0.04),
        object_mass_range_kg=(0.05, 0.05),
        support_mass_range_kg=(0.05, 0.05),
    )
    bundle = ActionBundle(
        ModelDescriptor("test-cube-bundle", "1", "action-bundle"),
        domain,
        policies,
    )
    return ActionService(ActionRouter({"cube": bundle}))
