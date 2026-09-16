from __future__ import annotations

from pathlib import Path
import sys

import torch

from stage_vla_v7.action.safety import (
    carrying_grasp_lost,
    entrance_action_scale,
    hold_finished_skill_action,
    jaw_leveling_axis_angle,
    object_upright_tilt_rad,
    project_align_motion,
    project_descend_motion,
    project_pregrasp_edge_alignment,
    project_skill_action,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.skill_action_safety import (  # noqa: E402
    carrying_grasp_lost as v5_carrying_grasp_lost,
    entrance_action_scale as v5_entrance_action_scale,
    hold_finished_skill_action as v5_hold_finished_skill_action,
    jaw_leveling_axis_angle as v5_jaw_leveling_axis_angle,
    object_upright_tilt_rad as v5_object_upright_tilt_rad,
    project_align_motion as v5_project_align_motion,
    project_descend_motion as v5_project_descend_motion,
    project_pregrasp_edge_alignment as v5_project_pregrasp_edge_alignment,
    project_skill_action as v5_project_skill_action,
)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    if actual.dtype == torch.bool:
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_skill_projection_and_finished_hold_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(1997)
    action = 2.5 * torch.randn(32, 5, generator=generator)
    previous = torch.randn(32, 5, generator=generator)
    finished = torch.rand(32, generator=generator) > 0.5
    for skill in (
        "REACH",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "ALIGN",
        "DESCEND",
        "RELEASE_STABILIZE",
        "RETREAT",
    ):
        _assert_exact(
            project_skill_action(skill, action),
            v5_project_skill_action(skill, action),
        )
        _assert_exact(
            hold_finished_skill_action(skill, action, finished, previous),
            v5_hold_finished_skill_action(skill, action, finished, previous),
        )


def test_geometry_and_warmup_safety_helpers_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(23)
    ages = torch.arange(32)
    _assert_exact(
        entrance_action_scale(ages, warmup_steps=17, start_scale=0.2),
        v5_entrance_action_scale(ages, warmup_steps=17, start_scale=0.2),
    )
    masks = [torch.rand(32, generator=generator) > 0.5 for _ in range(3)]
    _assert_exact(carrying_grasp_lost(*masks), v5_carrying_grasp_lost(*masks))
    left = torch.randn(32, 3, generator=generator)
    right = left + torch.randn(32, 3, generator=generator)
    right[:, :2] += torch.tensor([0.2, 0.1])
    _assert_exact(
        jaw_leveling_axis_angle(left, right),
        v5_jaw_leveling_axis_angle(left, right),
    )
    quaternion = torch.randn(32, 4, generator=generator)
    _assert_exact(
        object_upright_tilt_rad(quaternion),
        v5_object_upright_tilt_rad(quaternion),
    )


def test_motion_projection_helpers_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(71)
    action = torch.randn(48, 5, generator=generator)
    yaw_error = 0.1 * torch.randn(48, generator=generator)
    contact_free = torch.rand(48, generator=generator) > 0.4
    actual, actual_mask = project_pregrasp_edge_alignment(
        action, yaw_error, contact_free
    )
    expected, expected_mask = v5_project_pregrasp_edge_alignment(
        action, yaw_error, contact_free
    )
    _assert_exact(actual, expected)
    _assert_exact(actual_mask, expected_mask)

    relative = 0.03 * torch.randn(48, 3, generator=generator)
    target = 0.05 + 0.01 * torch.rand(48, generator=generator)
    _assert_exact(
        project_align_motion(
            action,
            relative,
            target,
            translation_limit_m=0.005,
            planar_height_margin_m=0.005,
        ),
        v5_project_align_motion(
            action,
            relative,
            target,
            translation_limit_m=0.005,
            planar_height_margin_m=0.005,
        ),
    )
    _assert_exact(
        project_descend_motion(
            action,
            relative,
            target,
            xy_tolerance_m=0.01,
            inner_xy_m=0.0075,
            translation_limit_m=0.003,
        ),
        v5_project_descend_motion(
            action,
            relative,
            target,
            xy_tolerance_m=0.01,
            inner_xy_m=0.0075,
            translation_limit_m=0.003,
        ),
    )
