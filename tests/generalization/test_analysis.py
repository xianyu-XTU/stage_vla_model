from __future__ import annotations

import pytest

from tools.generalization.analysis import aggregate_cases, extract_batch_cases, wilson_interval
from tools.generalization.manifest import generate_layout_manifest
from tools.generalization.run_pilot import _render_v2_report


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


def test_strict_vision_exception_overrides_stale_seed_valid() -> None:
    manifest = generate_layout_manifest(seed=19, layout_count=2)
    result = {
        "status": "failed",
        "failure": {
            "stage": "VISION",
            "failed_environments": [1],
        },
        "v7_chain": {"verified": False},
        "runtime_purity": {"verified": True},
        "vision": {
            "strict_mode": True,
            "seed_valid": [True, True],
            "invalid_frames": 1,
            "oracle_fallback_count": 0,
        },
    }
    cases = extract_batch_cases(result, manifest, result_path="vision.json")
    assert cases[0]["vision_valid"] is True
    assert cases[0]["first_failure_skill"] == "RUNTIME_ERROR"
    assert cases[1]["vision_valid"] is False
    assert cases[1]["first_failure_skill"] == "VISION"


def test_per_environment_outcomes_prevent_peer_abort_classification() -> None:
    manifest = generate_layout_manifest(seed=29, layout_count=2)
    result = _batch_result()
    result["environment_outcomes"] = [
        {
            "environment_index": 0,
            "outcome": "PASS",
            "physical_success": True,
            "stable_success": True,
            "first_failure_skill": None,
            "vision_valid": True,
            "vision_invalid_frames": 0,
            "vision_service_calls": 10,
            "v7_chain_verified": True,
            "peer_aborted_due_to_other_env_failure": False,
        },
        {
            "environment_index": 1,
            "outcome": "VISION_FAILURE",
            "physical_success": False,
            "stable_success": False,
            "first_failure_skill": "VISION",
            "failure_reason": "invalid_required_detection",
            "vision_valid": False,
            "vision_invalid_frames": 1,
            "vision_service_calls": 1,
            "v7_chain_verified": False,
            "peer_aborted_due_to_other_env_failure": False,
        },
    ]

    cases = extract_batch_cases(result, manifest, result_path="isolated.json")
    assert cases[0]["outcome"] == "PASS"
    assert cases[1]["outcome"] == "VISION_FAILURE"
    assert cases[1]["first_failure_skill"] == "VISION"
    assert cases[1]["failure_reason"] == "invalid_required_detection"
    assert not any(
        case["peer_aborted_due_to_other_env_failure"] for case in cases
    )

    aggregate = aggregate_cases(cases)
    assert aggregate["peer_aborted_due_to_other_env_failure"] == 0
    assert aggregate["outcome_taxonomy"]["VISION_FAILURE"] == 1
    assert aggregate["outcome_taxonomy"]["RUNTIME_ERROR"] == 0
    assert aggregate["outcome_taxonomy"]["GLOBAL_RUNTIME_ERROR"] == 0


def test_v2_report_contains_isolation_and_v1_comparison(tmp_path) -> None:
    manifest = generate_layout_manifest(seed=31, layout_count=2)
    cases = extract_batch_cases(_batch_result(), manifest, result_path="batch.json")
    aggregate = aggregate_cases(cases)
    aggregate["replays"] = []
    report = _render_v2_report(
        aggregate=aggregate,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest,
        batch_size=4,
        replays=(),
        source_commit="abc123",
        source_clean_before_run=True,
        checkpoint_hashes={"REACH": {"sha256": "ABC"}},
        v1_aggregate=aggregate,
        infrastructure_pass=True,
        ready_for_50=False,
    )

    assert "Pilot V2" in report
    assert "## V1 vs V2" in report
    assert "Peer-aborted cases" in report
    assert "GLOBAL_RUNTIME_ERROR" in report
    assert "READY_FOR_PHASE4_50               false" in report
