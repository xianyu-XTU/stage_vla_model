from __future__ import annotations

from pathlib import Path
import sys

from stage_vla_v7.simulation.isaac_lab.gripper_action import (
    KnownSizeGraspAction,
    KnownSizeGraspActionCfg,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.envs.known_size_grasp_action import (  # noqa: E402
    KnownSizeGraspAction as V5KnownSizeGraspAction,
)
from stage_vla.envs.known_size_grasp_action import (  # noqa: E402
    KnownSizeGraspActionCfg as V5KnownSizeGraspActionCfg,
)


def _cfg(cls):
    return cls(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        width_m=0.04,
        depth_m=0.04,
        height_m=0.04,
        mass_kg=0.05,
    )


def test_gripper_action_configuration_contract_matches_v5() -> None:
    actual = _cfg(KnownSizeGraspActionCfg)
    expected = _cfg(V5KnownSizeGraspActionCfg)
    for name in (
        "asset_name",
        "joint_names",
        "width_m",
        "depth_m",
        "height_m",
        "mass_kg",
        "grasp_width_ratio",
        "jaw_clearance_m",
        "max_compression_m",
        "friction_coefficient",
        "safety_factor",
        "min_force_n",
        "max_force_n",
        "pressure_tolerance_n",
        "force_balance_tolerance_n",
        "lift_acceleration_mps2",
        "joint_min_m",
        "joint_max_m",
        "open_position_m",
        "left_sensor_name",
        "right_sensor_name",
        "gain_m_per_n",
        "max_step_m",
        "residual_force_range_n",
        "residual_deadband",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def test_gripper_action_public_runtime_surface_matches_v5() -> None:
    for name in (
        "action_dim",
        "raw_actions",
        "processed_actions",
        "target_force_n",
        "measured_force_n",
        "oracle_config",
        "physical_batch",
        "geometry_adapter",
        "load_adapter",
        "set_physical_batch",
        "pressure_tracking_ok",
        "set_position_hold_mask",
        "process_actions",
        "apply_actions",
        "reset",
        "prime_closed",
    ):
        assert hasattr(KnownSizeGraspAction, name)
        assert hasattr(V5KnownSizeGraspAction, name)
