from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import torch

from stage_vla_v7.simulation.config import (
    KnownSizeGraspConfig,
    load_known_size_config,
    load_object_metadata,
)
from stage_vla_v7.simulation.isaac_lab.action_adapter import known_size_raw_action
from stage_vla_v7.simulation.physics import (
    grasp_contact_error,
    pressure_feedback_step,
    pressure_target_from_grip,
    pressure_tracking_ok,
    quaternion_control_angular_speed,
    size_conditioned_grasp_targets,
    stability_speed_from_source,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.known_size_grasp import (  # noqa: E402
    KnownSizeGraspConfig as V5KnownSizeGraspConfig,
)
from stage_vla.rl.known_size_grasp import (  # noqa: E402
    grasp_contact_error as v5_grasp_contact_error,
    known_size_raw_action as v5_known_size_raw_action,
    pressure_feedback_step as v5_pressure_feedback_step,
    pressure_target_from_grip as v5_pressure_target_from_grip,
    pressure_tracking_ok as v5_pressure_tracking_ok,
    quaternion_control_angular_speed as v5_quaternion_control_angular_speed,
    size_conditioned_grasp_targets as v5_size_conditioned_grasp_targets,
    stability_speed_from_source as v5_stability_speed_from_source,
)
from tools.train_known_size_grasp import (  # noqa: E402
    _load_object_metadata as v5_load_object_metadata,
    _load_size as v5_load_known_size_config,
)


def _configs() -> tuple[KnownSizeGraspConfig, V5KnownSizeGraspConfig]:
    values = dict(
        width_m=0.04,
        depth_m=0.04,
        height_m=0.04,
        mass_kg=0.05,
        grasp_width_m=0.032,
        jaw_clearance_m=0.002,
        max_compression_m=0.004,
        friction_coefficient=0.4,
        safety_factor=2.0,
        min_force_n=5.0,
        max_force_n=40.0,
        pressure_tolerance_n=1.0,
        force_balance_tolerance_n=1.5,
        residual_force_range_n=4.0,
        residual_action_penalty=0.5,
        lift_acceleration_mps2=0.5,
        joint_min_m=0.0,
        joint_max_m=0.04,
    )
    return KnownSizeGraspConfig(**values), V5KnownSizeGraspConfig(**values)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    if actual.dtype == torch.bool:
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_known_size_config_and_json_loader_match_v5() -> None:
    cfg, legacy = _configs()
    assert asdict(cfg) == asdict(legacy)
    assert cfg.geometric_joint_target_m == legacy.geometric_joint_target_m
    assert cfg.compression_joint_min_m == legacy.compression_joint_min_m
    assert cfg.initial_force_target_n == legacy.initial_force_target_n
    _assert_exact(cfg.size_features(), legacy.size_features())

    path = V5_ROOT / "config" / "v5_generalized_cube_eval.json"
    actual_loaded = load_known_size_config(path)
    expected_loaded = v5_load_known_size_config(path)
    assert asdict(actual_loaded) == asdict(expected_loaded)
    assert load_object_metadata(path) == v5_load_object_metadata(path)


def test_pressure_control_helpers_match_v5_exactly() -> None:
    cfg, legacy = _configs()
    generator = torch.Generator().manual_seed(61081)
    current = 0.012 + 0.025 * torch.rand(64, 2, generator=generator)
    measured = 8.0 * torch.rand(64, 2, generator=generator)
    grip = 2.5 * torch.rand(64, generator=generator) - 1.25
    target = 5.0 + 4.0 * torch.rand(64, generator=generator)
    _assert_exact(
        pressure_feedback_step(current, measured, target, cfg=cfg),
        v5_pressure_feedback_step(current, measured, target, cfg=legacy),
    )
    _assert_exact(
        pressure_target_from_grip(grip, cfg=cfg),
        v5_pressure_target_from_grip(grip, cfg=legacy),
    )
    _assert_exact(
        pressure_tracking_ok(measured, target, cfg=cfg),
        v5_pressure_tracking_ok(measured, target, cfg=legacy),
    )
    actual_joint, actual_force = size_conditioned_grasp_targets(cfg)
    expected_joint, expected_force = v5_size_conditioned_grasp_targets(legacy)
    _assert_exact(actual_joint, expected_joint)
    _assert_exact(actual_force, expected_force)


def test_motion_and_action_conversion_helpers_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(47)
    before = torch.randn(64, 4, generator=generator)
    after = before + 0.01 * torch.randn(64, 4, generator=generator)
    _assert_exact(
        quaternion_control_angular_speed(before, after, step_dt_s=0.05),
        v5_quaternion_control_angular_speed(before, after, step_dt_s=0.05),
    )
    instantaneous = torch.rand(64, generator=generator)
    control = torch.rand(64, generator=generator)
    for source in ("instantaneous", "control_delta"):
        _assert_exact(
            stability_speed_from_source(instantaneous, control, source),
            v5_stability_speed_from_source(instantaneous, control, source),
        )
    target = torch.randn(64, 3, generator=generator)
    left = torch.randn(64, 3, generator=generator)
    right = torch.randn(64, 3, generator=generator)
    _assert_exact(
        grasp_contact_error(target, left, right),
        v5_grasp_contact_error(target, left, right),
    )
    policy = 3.0 * torch.randn(64, 5, generator=generator)
    actual_raw, actual_unit = known_size_raw_action(policy)
    expected_raw, expected_unit = v5_known_size_raw_action(policy)
    _assert_exact(actual_raw, expected_raw)
    _assert_exact(actual_unit, expected_unit)
