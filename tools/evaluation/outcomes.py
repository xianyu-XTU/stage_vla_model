"""Build independent per-environment evaluation outcomes."""

from __future__ import annotations

from typing import Mapping, Sequence


SKILL_SEQUENCE = (
    "REACH",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "ALIGN",
    "DESCEND",
    "RELEASE_STABILIZE",
    "RETREAT",
)


def _as_bool_rows(value: object, count: int) -> list[bool]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    rows = list(value) if isinstance(value, Sequence) else []
    return [bool(rows[index]) if index < len(rows) else False for index in range(count)]


def _skill_row(
    relation: Mapping[str, object], skill: str, environment_index: int
) -> Mapping[str, object] | None:
    rows_by_skill = relation.get("rows", {})
    if not isinstance(rows_by_skill, Mapping):
        return None
    rows = rows_by_skill.get(skill, ())
    if not isinstance(rows, Sequence):
        return None
    return next(
        (
            row for row in rows
            if isinstance(row, Mapping)
            and int(row.get("env", -1)) == environment_index
        ),
        None,
    )


def build_environment_outcomes(
    *,
    num_envs: int,
    relation_results: Sequence[Mapping[str, object]],
    overall_alive: object,
    vision: Mapping[str, object],
    runtime_purity: Mapping[str, object],
    pipeline_audit: Mapping[str, object],
    reference_skill_calls: int,
    recovery_calls: int,
) -> list[dict[str, object]]:
    """Classify each vectorized environment without peer-failure propagation."""
    alive = _as_bool_rows(overall_alive, num_envs)
    per_environment = vision.get("per_environment", ())
    vision_rows = {
        int(row["environment_index"]): row
        for row in per_environment
        if isinstance(row, Mapping) and "environment_index" in row
    } if isinstance(per_environment, Sequence) else {}
    service_calls = vision.get("v7_service_calls_by_environment", ())
    service_calls = service_calls if isinstance(service_calls, Sequence) else ()
    outcomes: list[dict[str, object]] = []
    for environment_index in range(num_envs):
        vision_row = vision_rows.get(environment_index, {})
        vision_valid = bool(vision_row.get("valid", True))
        first_failure = None
        outcome = "PASS" if alive[environment_index] else "FINAL_STABILITY"
        failure_reason = None
        failure_telemetry = None
        if not vision_valid:
            outcome = "VISION_FAILURE"
            first_failure = "VISION"
            failure_reason = vision_row.get("failure_reason")
        elif not alive[environment_index]:
            for relation in relation_results:
                grasp = _skill_row(relation, "GRASP", environment_index)
                if grasp is None:
                    first_failure = "REACH"
                    outcome = "SKILL_FAILURE"
                    break
                for skill in SKILL_SEQUENCE[1:]:
                    row = _skill_row(relation, skill, environment_index)
                    if row is not None and not bool(row.get("success")):
                        first_failure = skill
                        failure_telemetry = dict(row)
                        outcome = "TIMEOUT" if bool(row.get("timeout")) else "SKILL_FAILURE"
                        break
                if first_failure is not None:
                    break
            if first_failure is None:
                first_failure = "FINAL_STABILITY"

        successful = alive[environment_index] and vision_valid
        calls = (
            int(service_calls[environment_index])
            if environment_index < len(service_calls)
            else 0
        )
        per_env_v7 = bool(
            vision.get("strict_mode") is True
            and vision_valid
            and calls > 0
            and int(vision_row.get("invalid_frames", 0)) == 0
            and int(vision.get("oracle_fallback_count", 0)) == 0
            and int(reference_skill_calls) == 0
            and int(recovery_calls) == 0
            and pipeline_audit.get("all_prepared_skills_exercised") is True
            and runtime_purity.get("vendor_path_exposed") is False
            and int(runtime_purity.get("loaded_v5_module_count", 1)) == 0
            and runtime_purity.get("import_blocker_enabled") is True
            and runtime_purity.get("verified") is True
        )
        outcomes.append({
            "environment_index": environment_index,
            "outcome": outcome,
            "physical_success": successful,
            "stable_success": successful,
            "first_failure_skill": first_failure,
            "failure_reason": failure_reason,
            "failure_telemetry": failure_telemetry,
            "vision_valid": vision_valid,
            "vision_invalid_frames": int(vision_row.get("invalid_frames", 0)),
            "vision_first_invalid_step": vision_row.get("first_invalid_step"),
            "vision_failed_objects": list(vision_row.get("failed_objects", ())),
            "vision_service_calls": calls,
            "v7_chain_verified": per_env_v7,
            "peer_aborted_due_to_other_env_failure": False,
        })
    return outcomes


def build_global_error_outcomes(
    num_envs: int, failure: Mapping[str, object]
) -> list[dict[str, object]]:
    """Account for every environment when a true process-wide error aborts a batch."""
    reason = str(failure.get("message", failure.get("reason", "global runtime error")))
    return [
        {
            "environment_index": index,
            "outcome": "GLOBAL_RUNTIME_ERROR",
            "physical_success": False,
            "stable_success": False,
            "first_failure_skill": "GLOBAL_RUNTIME_ERROR",
            "failure_reason": reason,
            "vision_valid": None,
            "v7_chain_verified": False,
            "peer_aborted_due_to_other_env_failure": False,
        }
        for index in range(num_envs)
    ]


__all__ = ["build_environment_outcomes", "build_global_error_outcomes"]
