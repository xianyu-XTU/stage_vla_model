"""Portable JSON reporting helpers for the M10 D1 evaluator.

This module is deliberately Isaac-free.  It serializes the already-computed
strict M7 and GRIP-preference summaries so checkpoint evaluations can be
compared without copying numbers by hand.  It does not change training,
actions, rewards, policy inference, or success semantics.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

D1_REPORT_SCHEMA = "stage_vla.m10_d1.v1"


def _summary_dict(value: object, *, name: str) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError(f"{name} must be a dataclass summary")
    data = asdict(value)
    if not isinstance(data, dict):
        raise TypeError(f"{name} did not serialize to a mapping")
    return data


def build_m10_d1_report(
    *,
    checkpoint: str | Path,
    project_version: str,
    seed: int,
    num_envs: int,
    steps: int,
    rsl_rl_version: str,
    strict_summary: object,
    grip_summary: object,
) -> dict[str, Any]:
    """Build one machine-readable D1 checkpoint report."""
    if int(num_envs) <= 0 or int(steps) <= 0:
        raise ValueError("num_envs and steps must be > 0")
    checkpoint_path = Path(checkpoint)
    strict_data = _summary_dict(strict_summary, name="strict_summary")
    grip_data = _summary_dict(grip_summary, name="grip_summary")
    if hasattr(grip_summary, "open_argmax_fraction"):
        grip_data["open_argmax_fraction"] = float(getattr(grip_summary, "open_argmax_fraction"))
    return {
        "schema": D1_REPORT_SCHEMA,
        "project_version": str(project_version),
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "seed": int(seed),
        "num_envs": int(num_envs),
        "steps": int(steps),
        "rsl_rl_version": str(rsl_rl_version),
        "strict_m7": strict_data,
        "grip_preference": grip_data,
    }


def write_m10_d1_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write a D1 report as deterministic UTF-8 JSON and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    validate_m10_d1_report(payload)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_m10_d1_report(path: str | Path) -> dict[str, Any]:
    """Load and validate one D1 JSON report."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"D1 report must contain a JSON object: {source}")
    validate_m10_d1_report(payload)
    return payload


def validate_m10_d1_report(report: Mapping[str, Any]) -> None:
    """Validate only fields required by the portable comparison workflow."""
    if report.get("schema") != D1_REPORT_SCHEMA:
        raise ValueError(
            f"unsupported D1 report schema {report.get('schema')!r}; expected {D1_REPORT_SCHEMA!r}"
        )
    for key in (
        "project_version",
        "checkpoint",
        "checkpoint_name",
        "seed",
        "num_envs",
        "steps",
        "rsl_rl_version",
        "strict_m7",
        "grip_preference",
    ):
        if key not in report:
            raise ValueError(f"missing D1 report field: {key}")
    if int(report["num_envs"]) <= 0 or int(report["steps"]) <= 0:
        raise ValueError("D1 report num_envs and steps must be > 0")
    if not isinstance(report["strict_m7"], Mapping):
        raise ValueError("strict_m7 must be a mapping")
    if not isinstance(report["grip_preference"], Mapping):
        raise ValueError("grip_preference must be a mapping")


def assert_d1_pair_compatible(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    allow_project_version_mismatch: bool = False,
) -> None:
    """Require the two reports to use the same controlled evaluation budget.

    Cross-version comparison is disabled by default.  It may be explicitly
    enabled for a preregistered same-evaluator comparison such as M10-5
    v0.10.6 versus M10-6-S1 v0.10.7, where only training reward strength
    changed and seed/env-count/evaluation horizon remain controlled.
    """
    validate_m10_d1_report(a)
    validate_m10_d1_report(b)
    mismatches: list[str] = []
    keys = ["seed", "num_envs", "steps"]
    if not allow_project_version_mismatch:
        keys.insert(0, "project_version")
    for key in keys:
        if a[key] != b[key]:
            mismatches.append(f"{key}: {a[key]!r} != {b[key]!r}")
    if mismatches:
        raise ValueError("incompatible D1 reports: " + "; ".join(mismatches))


def d1_comparison_rows(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    allow_project_version_mismatch: bool = False,
) -> list[tuple[str, Any, Any]]:
    """Return the pre-registered load-bearing metrics for side-by-side display."""
    assert_d1_pair_compatible(
        a, b, allow_project_version_mismatch=allow_project_version_mismatch
    )
    sa, sb = a["strict_m7"], b["strict_m7"]
    ga, gb = a["grip_preference"], b["grip_preference"]
    return [
        ("physical grasp", sa["ever_physical_grasp"], sb["ever_physical_grasp"]),
        ("stable grasp", sa["ever_stable_grasp"], sb["ever_stable_grasp"]),
        ("lifted", sa["ever_lifted"], sb["ever_lifted"]),
        ("red-on-blue geometry", sa["ever_geometry_ok"], sb["ever_geometry_ok"]),
        ("post-lift gripper open", sa["ever_postlift_gripper_open"], sb["ever_postlift_gripper_open"]),
        ("post-lift released", sa["ever_postlift_released"], sb["ever_postlift_released"]),
        ("geometry + released", sa["ever_geometry_released"], sb["ever_geometry_released"]),
        ("strict success", sa["strict_successes"], sb["strict_successes"]),
        ("strict success rate", sa.get("strict_success_rate", 0.0), sb.get("strict_success_rate", 0.0)),
        ("release-ready samples", ga["ready_samples"], gb["ready_samples"]),
        ("release-ready envs", ga["ready_envs"], gb["ready_envs"]),
        ("OPEN argmax samples", ga["argmax_open_samples"], gb["argmax_open_samples"]),
        ("KEEP argmax samples", ga["argmax_keep_samples"], gb["argmax_keep_samples"]),
        ("CLOSE argmax samples", ga["argmax_close_samples"], gb["argmax_close_samples"]),
        ("OPEN argmax envs", ga["argmax_open_envs"], gb["argmax_open_envs"]),
        ("OPEN argmax fraction", ga.get("open_argmax_fraction", 0.0), gb.get("open_argmax_fraction", 0.0)),
        ("mean P(OPEN)", ga["mean_open_prob"], gb["mean_open_prob"]),
        ("mean P(KEEP)", ga["mean_keep_prob"], gb["mean_keep_prob"]),
        ("mean P(CLOSE)", ga["mean_close_prob"], gb["mean_close_prob"]),
        ("max P(OPEN)", ga["max_open_prob"], gb["max_open_prob"]),
        ("mean OPEN margin", ga["mean_open_margin"], gb["mean_open_margin"]),
        ("max OPEN margin", ga["max_open_margin"], gb["max_open_margin"]),
    ]
