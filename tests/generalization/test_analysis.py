from __future__ import annotations

import pytest

from tools.generalization.analysis import aggregate_cases, extract_batch_cases, wilson_interval
from tools.generalization.manifest import generate_layout_manifest


def _successful_row(env: int, *, success: bool = True) -> dict[str, object]:
    return {
        "env": env,
        "steps": 10,
        "success": success,
        "failure": not success,
        "timeout": False,
        "physical_grasp": success,
        "gripper_joint_m": [0.02, 0.02],
        "stack_xy_m": 0.01,
        "stack_relative_height_m": 0.0468,
    }


def _batch_result() -> dict[str, object]:
    rows = {
        "GRASP": [_successful_row(0), _successful_row(1, success=False)],
        "LIFT": [_successful_row(0)],
        "TRANSPORT": [_successful_row(0)],
        "ALIGN": [_successful_row(0)],
        "DESCEND": [_successful_row(0)],
        "RELEASE_STABILIZE": [_successful_row(0)],
        "RETREAT": [_successful_row(0)],
    }
    return {
        "status": "failed",
        "v7_chain": {"verified": True},
        "runtime_purity": {
            "verified": True,
            "vendor_path_exposed": False,
            "loaded_v5_module_count": 0,
        },
        "closeout": {"reference_skill_calls": 0, "recovery_calls": 0},
        "vision": {
            "strict_mode": True,
            "seed_valid": [True, True],
            "v7_service_calls": 20,
            "invalid_frames": 0,
            "oracle_fallback_count": 0,
        },
        "seed_success": [True, False],
        "thresholds": {
            "align_xy_m": 0.01,
            "align_height_m": 0.0618,
            "align_height_tolerance_m": 0.015,
        },
        "relations": [{
            "reach_steps": [12, 13],
            "rows": rows,
            "env_success": [True, False],
        }],
    }


def test_extracts_skill_funnel_and_first_failure() -> None:
    manifest = generate_layout_manifest(seed=9, layout_count=2)
    cases = extract_batch_cases(_batch_result(), manifest, result_path="batch.json")
    assert cases[0]["stable_success"] is True
    assert cases[0]["first_failure_skill"] is None
    assert cases[1]["stable_success"] is False
    assert cases[1]["first_failure_skill"] == "GRASP"
    assert cases[1]["failure_type"] == "GRASP_MISS"
    assert cases[1]["skills"]["REACH"]["success"] is True
    assert cases[1]["skills"]["LIFT"]["entered"] is False

    aggregate = aggregate_cases(cases)
    assert aggregate["physical_success_count"] == 1
    assert aggregate["skill_funnel"]["REACH"] == {
        "entered": 2,
        "success": 2,
        "conditional_success_rate": 1.0,
    }
    assert aggregate["skill_funnel"]["GRASP"] == {
        "entered": 2,
        "success": 1,
        "conditional_success_rate": 0.5,
    }
    assert aggregate["skill_funnel"]["LIFT"]["entered"] == 1
    assert aggregate["first_failure_distribution"]["GRASP"] == 1


def test_wilson_interval_has_expected_bounds() -> None:
    low, high = wilson_interval(9, 20)
    assert low == pytest.approx(0.2582, abs=1e-4)
    assert high == pytest.approx(0.6579, abs=1e-4)
