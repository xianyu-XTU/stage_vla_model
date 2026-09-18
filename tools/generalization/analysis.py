"""Aggregate per-environment evaluator telemetry for the Phase 4 pilot."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


SKILLS = (
    "REACH",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "ALIGN",
    "DESCEND",
    "RELEASE_STABILIZE",
    "RETREAT",
)
FIRST_FAILURE_KEYS = (*SKILLS, "VISION", "FINAL_STABILITY", "RUNTIME_ERROR", "UNKNOWN")
FAILURE_TYPES = (
    "VISION_FAILURE",
    "REACH_GEOMETRY",
    "GRASP_MISS",
    "GRASP_UNSTABLE",
    "LIFT_DROP",
    "TRANSPORT_DROP",
    "ALIGN_XY",
    "ALIGN_HEIGHT",
    "DESCEND_GEOMETRY",
    "RELEASE_NOT_OPEN",
    "RELEASE_DISTURBANCE",
    "RETREAT_COLLISION",
    "FINAL_STACK_UNSTABLE",
    "TIMEOUT",
    "RUNTIME_ERROR",
    "UNKNOWN",
)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def _row_map(relation: Mapping[str, object], skill: str) -> dict[int, dict[str, object]]:
    rows_by_skill = relation.get("rows", {})
    if not isinstance(rows_by_skill, Mapping):
        return {}
    rows = rows_by_skill.get(skill, ())
    if not isinstance(rows, Sequence):
        return {}
    return {
        int(row["env"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and "env" in row
    }


def _failure_type(
    first_failure: str | None,
    row: Mapping[str, object] | None,
    thresholds: Mapping[str, object],
) -> str | None:
    if first_failure is None:
        return None
    if first_failure == "VISION":
        return "VISION_FAILURE"
    if first_failure == "RUNTIME_ERROR":
        return "RUNTIME_ERROR"
    if first_failure == "FINAL_STABILITY":
        return "FINAL_STACK_UNSTABLE"
    if first_failure == "REACH":
        return "REACH_GEOMETRY"
    if row is None:
        return "UNKNOWN"
    if bool(row.get("timeout")):
        return "TIMEOUT"
    if first_failure == "GRASP":
        return "GRASP_MISS" if not bool(row.get("physical_grasp")) else "GRASP_UNSTABLE"
    if first_failure == "LIFT":
        return "LIFT_DROP"
    if first_failure == "TRANSPORT":
        return "TRANSPORT_DROP"
    if first_failure == "ALIGN":
        xy = float(row.get("stack_xy_m", math.inf))
        relative_height = float(row.get("stack_relative_height_m", math.inf))
        target = float(thresholds.get("align_height_m", 0.0618))
        tolerance = float(thresholds.get("align_height_tolerance_m", 0.015))
        if xy > float(thresholds.get("align_xy_m", 0.010)):
            return "ALIGN_XY"
        if abs(relative_height - target) > tolerance:
            return "ALIGN_HEIGHT"
        return "UNKNOWN"
    if first_failure == "DESCEND":
        return "DESCEND_GEOMETRY"
    if first_failure == "RELEASE_STABILIZE":
        joints = row.get("gripper_joint_m", ())
        if isinstance(joints, Sequence) and joints and min(float(value) for value in joints) < 0.03:
            return "RELEASE_NOT_OPEN"
        return "RELEASE_DISTURBANCE"
    if first_failure == "RETREAT":
        return "RETREAT_COLLISION"
    return "UNKNOWN"


def _geometry(object_xyz: Sequence[float], support_xyz: Sequence[float]) -> dict[str, float]:
    dx = float(object_xyz[0]) - float(support_xyz[0])
    dy = float(object_xyz[1]) - float(support_xyz[1])
    return {
        "object_x": float(object_xyz[0]),
        "object_y": float(object_xyz[1]),
        "support_x": float(support_xyz[0]),
        "support_y": float(support_xyz[1]),
        "relative_dx_m": dx,
        "relative_dy_m": dy,
        "xy_distance_m": math.hypot(dx, dy),
    }


def extract_batch_cases(
    result: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    result_path: str,
) -> list[dict[str, object]]:
    """Convert one evaluator batch result into layout-addressable case rows."""
    layout_ids = [str(value) for value in manifest["layout_ids"]]
    positions = manifest["asset_positions_local_xyz"]
    object_asset = str(manifest["object_asset"])
    support_asset = str(manifest["support_asset"])
    failure = result.get("failure", {})
    failure_stage = failure.get("stage") if isinstance(failure, Mapping) else None
    failed_vision_envs = {
        int(value) for value in failure.get("failed_environments", ())
    } if failure_stage == "VISION" else set()
    relations = result.get("relations", ())
    relation = relations[0] if isinstance(relations, Sequence) and relations else None
    relation = relation if isinstance(relation, Mapping) else None
    vision = result.get("vision", {})
    vision = vision if isinstance(vision, Mapping) else {}
    runtime_purity = result.get("runtime_purity", {})
    runtime_purity = runtime_purity if isinstance(runtime_purity, Mapping) else {}
    v7_chain = result.get("v7_chain", {})
    v7_chain = v7_chain if isinstance(v7_chain, Mapping) else {}
    closeout = result.get("closeout", {})
    closeout = closeout if isinstance(closeout, Mapping) else {}
    thresholds = result.get("thresholds", {})
    thresholds = thresholds if isinstance(thresholds, Mapping) else {}
    seed_success = result.get("seed_success", ())
    seed_success = seed_success if isinstance(seed_success, Sequence) else ()
    seed_valid = vision.get("seed_valid", ())
    seed_valid = seed_valid if isinstance(seed_valid, Sequence) else ()
    service_calls = int(vision.get("v7_service_calls", 0))
    per_env_vision_calls = service_calls // len(layout_ids) if layout_ids else 0
    stage_rows = {
        skill: _row_map(relation, skill) if relation is not None else {}
        for skill in SKILLS[1:]
    }
    cases: list[dict[str, object]] = []
    for env_index, layout_id in enumerate(layout_ids):
        object_xyz = list(positions[object_asset][env_index])
        support_xyz = list(positions[support_asset][env_index])
        runtime_error = relation is None and env_index not in failed_vision_envs
        vision_valid = env_index not in failed_vision_envs and (
            bool(seed_valid[env_index])
            if env_index < len(seed_valid)
            else failure_stage != "VISION"
        )
        skill_state: dict[str, dict[str, object]] = {}
        reach_success = env_index in stage_rows["GRASP"]
        if relation is not None and not stage_rows["GRASP"]:
            relation_failure = relation.get("failure", {})
            if isinstance(relation_failure, Mapping) and relation_failure.get("stage") == "REACH":
                reach_values = relation.get("env_success", ())
                if isinstance(reach_values, Sequence) and env_index < len(reach_values):
                    reach_success = bool(reach_values[env_index])
        reach_steps = relation.get("reach_steps", ()) if relation is not None else ()
        reach_step_count = (
            int(reach_steps[env_index])
            if isinstance(reach_steps, Sequence) and env_index < len(reach_steps)
            else 0
        )
        skill_state["REACH"] = {
            "entered": relation is not None and vision_valid,
            "success": bool(reach_success),
            "steps": reach_step_count,
            "telemetry": None,
        }
        for skill in SKILLS[1:]:
            row = stage_rows[skill].get(env_index)
            skill_state[skill] = {
                "entered": row is not None,
                "success": bool(row and row.get("success")),
                "steps": int(row.get("steps", 0)) if row else 0,
                "telemetry": row,
            }
        physical_success = bool(
            env_index < len(seed_success) and seed_success[env_index]
        )
        stable_success = physical_success
        first_failure: str | None = None
        failed_row: Mapping[str, object] | None = None
        if env_index in failed_vision_envs:
            first_failure = "VISION"
        elif runtime_error:
            first_failure = "RUNTIME_ERROR"
        else:
            for skill in SKILLS:
                state = skill_state[skill]
                if bool(state["entered"]) and not bool(state["success"]):
                    first_failure = skill
                    telemetry = state["telemetry"]
                    failed_row = telemetry if isinstance(telemetry, Mapping) else None
                    break
            if first_failure is None and not stable_success:
                first_failure = "FINAL_STABILITY" if all(
                    bool(skill_state[skill]["success"]) for skill in SKILLS
                ) else "UNKNOWN"
        case = {
            "layout_id": layout_id,
            "batch_env_index": env_index,
            "object_asset": object_asset,
            "support_asset": support_asset,
            "object_xyz": object_xyz,
            "support_xyz": support_xyz,
            "geometry": _geometry(object_xyz, support_xyz),
            "physical_success": physical_success,
            "stable_success": stable_success,
            "v7_chain_verified": v7_chain.get("verified") is True,
            "runtime_purity_verified": runtime_purity.get("verified") is True,
            "vendor_path_exposed": runtime_purity.get("vendor_path_exposed"),
            "loaded_v5_module_count": runtime_purity.get("loaded_v5_module_count"),
            "strict_vision": vision.get("strict_mode") is True,
            "vision_valid": vision_valid,
            "vision_service_calls": per_env_vision_calls,
            "invalid_vision_frames": int(vision.get("invalid_frames", 0)),
            "oracle_fallback_count": int(vision.get("oracle_fallback_count", 0)),
            "reference_skill_calls": int(
                closeout.get(
                    "reference_skill_calls", result.get("reference_skill_calls", 0)
                )
            ),
            "recovery_calls": int(
                closeout.get("recovery_calls", result.get("recovery_calls", 0))
            ),
            "skills": skill_state,
            "episode_steps": {
                "by_skill": {
                    skill: int(skill_state[skill]["steps"]) for skill in SKILLS
                },
                "total_recorded": sum(
                    int(skill_state[skill]["steps"]) for skill in SKILLS
                ),
            },
            "first_failure_skill": first_failure,
            "failure_type": _failure_type(first_failure, failed_row, thresholds),
            "failure_reason": (
                "batch_aborted_by_peer_strict_vision_failure"
                if runtime_error and failure_stage == "VISION"
                else (
                    str(failure.get("message", "runtime exception aborted evaluation"))
                    if runtime_error and isinstance(failure, Mapping)
                    else None
                )
            ),
            "evaluation_result_path": result_path,
        }
        cases.append(case)
    return cases


def _bucket_summary(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[Mapping[str, object]]] = {
        "object_left": [],
        "object_right": [],
        "transport_near_lt_0.14m": [],
        "transport_far_ge_0.14m": [],
        "relative_dx_negative": [],
        "relative_dx_nonnegative": [],
        "relative_dy_negative": [],
        "relative_dy_nonnegative": [],
        "workspace_boundary_within_0.02m": [],
    }
    for case in cases:
        geometry = case["geometry"]
        object_x = float(geometry["object_x"])
        object_y = float(geometry["object_y"])
        distance = float(geometry["xy_distance_m"])
        buckets["object_left" if object_x < 0.5 else "object_right"].append(case)
        buckets[
            "transport_near_lt_0.14m" if distance < 0.14 else "transport_far_ge_0.14m"
        ].append(case)
        buckets[
            "relative_dx_negative"
            if float(geometry["relative_dx_m"]) < 0.0
            else "relative_dx_nonnegative"
        ].append(case)
        buckets[
            "relative_dy_negative"
            if float(geometry["relative_dy_m"]) < 0.0
            else "relative_dy_nonnegative"
        ].append(case)
        if min(object_x - 0.4, 0.6 - object_x, object_y + 0.1, 0.1 - object_y) <= 0.02:
            buckets["workspace_boundary_within_0.02m"].append(case)
    return {
        name: {
            "cases": len(rows),
            "stable_successes": sum(bool(row["stable_success"]) for row in rows),
            "failures": sum(not bool(row["stable_success"]) for row in rows),
        }
        for name, rows in buckets.items()
    }


def aggregate_cases(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(cases)
    if total < 1:
        raise ValueError("at least one case is required")
    physical = sum(bool(case["physical_success"]) for case in cases)
    stable = sum(bool(case["stable_success"]) for case in cases)
    skill_funnel = {}
    for skill in SKILLS:
        entered = sum(bool(case["skills"][skill]["entered"]) for case in cases)
        success = sum(bool(case["skills"][skill]["success"]) for case in cases)
        skill_funnel[skill] = {
            "entered": entered,
            "success": success,
            "conditional_success_rate": success / entered if entered else None,
        }
    first_failure = Counter(
        str(case["first_failure_skill"])
        for case in cases
        if case.get("first_failure_skill") is not None
    )
    failure_types = Counter(
        str(case["failure_type"])
        for case in cases
        if case.get("failure_type") is not None
    )
    physical_ci = wilson_interval(physical, total)
    stable_ci = wilson_interval(stable, total)
    return {
        "schema": "stage_vla_v7.phase4_pilot_aggregate.v1",
        "layout_count": total,
        "evaluated_cases": total,
        "physical_success_count": physical,
        "physical_success_rate": physical / total,
        "physical_success_wilson_95": list(physical_ci),
        "stable_success_count": stable,
        "stable_success_rate": stable / total,
        "stable_success_wilson_95": list(stable_ci),
        "v7_chain_valid_count": sum(bool(case["v7_chain_verified"]) for case in cases),
        "strict_vision_count": sum(bool(case["strict_vision"]) for case in cases),
        "vision_valid_count": sum(bool(case["vision_valid"]) for case in cases),
        "runtime_purity_valid_count": sum(
            bool(case["runtime_purity_verified"]) for case in cases
        ),
        "oracle_fallback_count": max(
            int(case["oracle_fallback_count"]) for case in cases
        ),
        "reference_skill_calls": max(
            int(case["reference_skill_calls"]) for case in cases
        ),
        "recovery_calls": max(int(case["recovery_calls"]) for case in cases),
        "skill_funnel": skill_funnel,
        "first_failure_distribution": {
            key: first_failure.get(key, 0) for key in FIRST_FAILURE_KEYS
        },
        "failure_taxonomy": {
            key: failure_types.get(key, 0) for key in FAILURE_TYPES
        },
        "position_conditioned": _bucket_summary(cases),
        "cases": list(cases),
    }


def write_json(path: Path, payload: Mapping[str, object] | Sequence[object]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "FAILURE_TYPES",
    "FIRST_FAILURE_KEYS",
    "SKILLS",
    "aggregate_cases",
    "extract_batch_cases",
    "wilson_interval",
    "write_json",
]
