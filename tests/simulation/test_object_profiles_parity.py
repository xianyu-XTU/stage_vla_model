from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import torch

from stage_vla_v7.simulation.config import KnownSizeGraspConfig
from stage_vla_v7.simulation.physics.object_profiles import (
    ACTION_CONTEXT_VERSION,
    GeometryAdapter,
    LoadAdapter,
    PhysicalDomainConfig,
    PhysicalObjectBatch,
    snapshot_has_physical_context,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.objects import ACTION_CONTEXT_VERSION as V5_ACTION_CONTEXT_VERSION  # noqa: E402
from stage_vla.rl.known_size_grasp import KnownSizeGraspConfig as V5KnownSizeGraspConfig  # noqa: E402
from stage_vla.rl.object_physics import GeometryAdapter as V5GeometryAdapter  # noqa: E402
from stage_vla.rl.object_physics import LoadAdapter as V5LoadAdapter  # noqa: E402
from stage_vla.rl.object_physics import PhysicalDomainConfig as V5PhysicalDomainConfig  # noqa: E402
from stage_vla.rl.object_physics import PhysicalObjectBatch as V5PhysicalObjectBatch  # noqa: E402
from stage_vla.rl.object_physics import (  # noqa: E402
    snapshot_has_physical_context as v5_snapshot_has_physical_context,
)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    if actual.dtype == torch.bool:
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def _assert_batch_parity(actual, expected) -> None:
    for name in (
        "object_size_m",
        "support_size_m",
        "object_mass_kg",
        "support_mass_kg",
        "object_deformable",
        "stackable",
    ):
        _assert_exact(getattr(actual, name), getattr(expected, name))
    assert actual.geometry_bundle == expected.geometry_bundle
    assert actual.context_version == expected.context_version
    assert actual.num_envs == expected.num_envs
    _assert_exact(actual.action_context(), expected.action_context())


def test_physical_domain_sampling_matches_v5_exactly() -> None:
    mapping = {
        "geometry_bundle": "box_parallel_jaw",
        "object_size_range_m": ((0.03, 0.05), (0.03, 0.05), (0.03, 0.05)),
        "support_size_range_m": ((0.04, 0.06), (0.04, 0.06), (0.03, 0.05)),
        "object_mass_range_kg": (0.03, 0.08),
        "support_mass_range_kg": (0.04, 0.10),
        "object_deformable": False,
        "stackable": True,
    }
    actual_cfg = PhysicalDomainConfig.from_mapping(mapping)
    expected_cfg = V5PhysicalDomainConfig.from_mapping(mapping)
    assert asdict(actual_cfg) == asdict(expected_cfg)
    _assert_batch_parity(
        actual_cfg.sample(64, seed=61081), expected_cfg.sample(64, seed=61081)
    )


def test_batch_constructors_and_snapshot_contract_match_v5_exactly() -> None:
    assert ACTION_CONTEXT_VERSION == V5_ACTION_CONTEXT_VERSION
    config_values = dict(width_m=0.04, depth_m=0.04, height_m=0.04, mass_kg=0.05)
    actual_cfg = KnownSizeGraspConfig(**config_values)
    expected_cfg = V5KnownSizeGraspConfig(**config_values)
    actual = PhysicalObjectBatch.from_scalar_config(actual_cfg, 7)
    expected = V5PhysicalObjectBatch.from_scalar_config(expected_cfg, 7)
    _assert_batch_parity(actual, expected)

    context = actual.action_context()
    _assert_batch_parity(
        PhysicalObjectBatch.from_action_context(
            context, geometry_bundle="box_parallel_jaw"
        ),
        V5PhysicalObjectBatch.from_action_context(
            context, geometry_bundle="box_parallel_jaw"
        ),
    )
    payloads = [
        {
            "physical_context_version": ACTION_CONTEXT_VERSION,
            "physical_context": row.tolist(),
            "geometry_bundle": "box_parallel_jaw",
        }
        for row in context[:3]
    ]
    assert all(
        snapshot_has_physical_context(payload)
        == v5_snapshot_has_physical_context(payload)
        for payload in payloads
    )
    _assert_batch_parity(
        PhysicalObjectBatch.from_snapshot_payloads(payloads, 8),
        V5PhysicalObjectBatch.from_snapshot_payloads(payloads, 8),
    )


def test_geometry_and_load_adapters_match_v5_exactly() -> None:
    mapping = {
        "geometry_bundle": "box_parallel_jaw",
        "object_size_range_m": ((0.03, 0.045),) * 3,
        "support_size_range_m": ((0.04, 0.055),) * 3,
        "object_mass_range_kg": (0.03, 0.08),
        "support_mass_range_kg": (0.04, 0.10),
    }
    actual_batch = PhysicalDomainConfig.from_mapping(mapping).sample(32, seed=19)
    expected_batch = V5PhysicalDomainConfig.from_mapping(mapping).sample(32, seed=19)
    actual_geometry = GeometryAdapter(actual_batch)
    expected_geometry = V5GeometryAdapter(expected_batch)
    for name in (
        "effective_width_m",
        "minimum_planar_width_m",
        "geometric_joint_target_m",
        "compression_joint_min_m",
        "stack_center_separation_m",
    ):
        _assert_exact(getattr(actual_geometry, name), getattr(expected_geometry, name))
    _assert_exact(
        actual_geometry.legacy_size_features(),
        expected_geometry.legacy_size_features(),
    )
    generator = torch.Generator().manual_seed(47)
    current = 0.01 + 0.03 * torch.rand(32, 2, generator=generator)
    force = 8.0 * torch.rand(32, 2, generator=generator)
    target = 5.0 + 4.0 * torch.rand(32, generator=generator)
    _assert_exact(
        actual_geometry.pressure_feedback_step(
            current, force, target, gain_m_per_n=0.0002, max_step_m=0.001
        ),
        expected_geometry.pressure_feedback_step(
            current, force, target, gain_m_per_n=0.0002, max_step_m=0.001
        ),
    )

    actual_load = LoadAdapter(actual_batch)
    expected_load = V5LoadAdapter(expected_batch)
    _assert_exact(actual_load.initial_force_target_n, expected_load.initial_force_target_n)
    grip = 2.0 * torch.rand(32, generator=generator) - 1.0
    _assert_exact(actual_load.target_from_grip(grip), expected_load.target_from_grip(grip))
    _assert_exact(
        actual_load.tracking_ok(force, target),
        expected_load.tracking_ok(force, target),
    )
