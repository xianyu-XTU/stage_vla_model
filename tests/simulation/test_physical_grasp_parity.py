from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys

import torch

from stage_vla_v7.simulation.physics import (
    GraspGeometryProfile,
    PhysicalGraspConfig,
    build_parallel_jaw_candidate,
    parallel_jaw_yaw_error,
    physical_grasp_diagnostics,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.stages.grasp_geometry import (  # noqa: E402
    GraspGeometryProfile as V5GraspGeometryProfile,
)
from stage_vla.stages.grasp_geometry import (  # noqa: E402
    build_parallel_jaw_candidate as v5_build_parallel_jaw_candidate,
)
from stage_vla.stages.grasp_geometry import (  # noqa: E402
    parallel_jaw_yaw_error as v5_parallel_jaw_yaw_error,
)
from stage_vla.stages.physical_grasp import (  # noqa: E402
    PhysicalGraspConfig as V5PhysicalGraspConfig,
)
from stage_vla.stages.physical_grasp import (  # noqa: E402
    physical_grasp_diagnostics as v5_physical_grasp_diagnostics,
)


def _assert_tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    if actual.dtype == torch.bool:
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_parallel_jaw_candidate_matches_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(61081)
    positions = torch.randn(32, 3, generator=generator)
    sizes = 0.02 + 0.06 * torch.rand(32, 3, generator=generator)
    for geometry, height_ratio, width_ratio in (
        ("box", 0.5, None),
        ("cylinder", 0.4, 0.8),
        ("cone", 0.25, None),
    ):
        profile = GraspGeometryProfile(geometry, height_ratio, width_ratio)
        expected_profile = V5GraspGeometryProfile(geometry, height_ratio, width_ratio)
        actual = build_parallel_jaw_candidate(positions, sizes, profile=profile)
        expected = v5_build_parallel_jaw_candidate(
            positions, sizes, profile=expected_profile
        )
        _assert_tensor_parity(actual.target_position_w, expected.target_position_w)
        _assert_tensor_parity(actual.effective_width_m, expected.effective_width_m)
        _assert_tensor_parity(actual.width_ratio, expected.width_ratio)
        assert actual.height_ratio == expected.height_ratio


def test_parallel_jaw_yaw_error_matches_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(1997)
    count = 64
    quaternion = torch.randn(count, 4, generator=generator)
    size = 0.02 + 0.06 * torch.rand(count, 3, generator=generator)
    left = torch.randn(count, 3, generator=generator)
    jaw = torch.randn(count, 3, generator=generator)
    jaw[:, :2] += torch.tensor([0.2, 0.1])
    right = left + jaw
    actual = parallel_jaw_yaw_error(quaternion, size, left, right)
    expected = v5_parallel_jaw_yaw_error(quaternion, size, left, right)
    _assert_tensor_parity(actual, expected)


def test_physical_grasp_diagnostics_matches_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(47)
    count = 128
    object_position = torch.randn(count, 3, generator=generator)
    left = object_position + 0.02 * torch.randn(count, 3, generator=generator)
    right = object_position + 0.02 * torch.randn(count, 3, generator=generator)
    right[:, 0] += 0.04
    force_a = 2.0 * torch.rand(count, generator=generator)
    force_b = 2.0 * torch.rand(count, generator=generator)
    reference = object_position + 0.005 * torch.randn(count, 3, generator=generator)
    tolerance = 0.006 + 0.01 * torch.rand(count, generator=generator)
    cfg = PhysicalGraspConfig(0.03, 0.012, 0.5, 0.05)
    v5_cfg = V5PhysicalGraspConfig(0.03, 0.012, 0.5, 0.05)
    actual = physical_grasp_diagnostics(
        object_position,
        left,
        right,
        force_a,
        force_b,
        cfg=cfg,
        reference_pos_w=reference,
        height_tolerance_m=tolerance,
    )
    expected = v5_physical_grasp_diagnostics(
        object_position,
        left,
        right,
        force_a,
        force_b,
        cfg=v5_cfg,
        reference_pos_w=reference,
        height_tolerance_m=tolerance,
    )
    assert [field.name for field in fields(actual)] == [
        field.name for field in fields(expected)
    ]
    for field in fields(actual):
        _assert_tensor_parity(
            getattr(actual, field.name), getattr(expected, field.name)
        )
