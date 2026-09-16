"""Acceptance gates and report aggregation for action-model candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CandidateValidationCase:
    """One independently launched Isaac validation case."""

    name: str
    seed: int
    num_envs: int
    layout: str
    layout_seed: int | None = None

    def validate(self) -> None:
        if not self.name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in self.name.lower()):
            raise ValueError(f"invalid case name: {self.name!r}")
        if self.num_envs < 1:
            raise ValueError("case num_envs must be positive")
        if self.layout not in ("fixed", "random_safe"):
            raise ValueError("case layout must be fixed or random_safe")
        if self.layout == "random_safe" and self.layout_seed is None:
            raise ValueError("random_safe cases require layout_seed")
        if self.layout == "fixed" and self.layout_seed is not None:
            raise ValueError("fixed cases must not define layout_seed")


def load_candidate_suite(path: Path) -> tuple[CandidateValidationCase, ...]:
    """Load a declarative candidate suite without importing Isaac Lab."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "stage_vla_v5.action_candidate_suite.v1":
        raise ValueError(f"unsupported candidate suite schema: {payload.get('schema')!r}")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate suite must define at least one case")
    cases = tuple(
        CandidateValidationCase(
            name=str(row["name"]),
            seed=int(row["seed"]),
            num_envs=int(row["num_envs"]),
            layout=str(row["layout"]),
            layout_seed=(int(row["layout_seed"]) if row.get("layout_seed") is not None else None),
        )
        for row in rows
    )
    for case in cases:
        case.validate()
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("candidate suite case names must be unique")
    return cases


def _flatten_ints(value: object) -> list[int]:
    if isinstance(value, list):
        output: list[int] = []
        for item in value:
            output.extend(_flatten_ints(item))
        return output
    return [int(value)]


def assess_candidate_result(
    result: Mapping[str, object],
    *,
    expected_episodes: int,
) -> tuple[str, ...]:
    """Return explicit rejection reasons; an empty tuple means the case passes."""
    reasons: list[str] = []
    if result.get("status") != "passed":
        reasons.append("result status is not passed")
    if int(result.get("episodes", -1)) != int(expected_episodes):
        reasons.append("episode count does not match the suite case")
    if int(result.get("chain_successes", -1)) != int(expected_episodes):
        reasons.append("not every episode completed the full chain")
    if result.get("pure_policy_actions") is not True:
        reasons.append("actions were not pure policy outputs")
    if int(result.get("mid_episode_resets", -1)) != 0:
        reasons.append("simulator reset occurred during an episode")
    if result.get("reach_recovery_controller") is not None:
        reasons.append("a REACH recovery controller was used")
    recovery = result.get("reach_recovery_steps_used", [])
    try:
        if any(value != 0 for value in _flatten_ints(recovery)):
            reasons.append("geometric REACH recovery steps were used")
    except (TypeError, ValueError):
        reasons.append("REACH recovery accounting is malformed")
    final_stack = result.get("final_stack")
    if not isinstance(final_stack, Mapping):
        reasons.append("final stack evidence is missing")
    else:
        if int(final_stack.get("successes", -1)) != int(expected_episodes):
            reasons.append("not every final stack passed")
        if final_stack.get("stability_speed_source") != "control_delta":
            reasons.append("final stability did not use control-delta speed")
    relations = result.get("relations")
    if not isinstance(relations, list) or len(relations) != 3:
        reasons.append("three relation results were not recorded")
    elif any(not isinstance(row, Mapping) or row.get("passed") is not True for row in relations):
        reasons.append("at least one stack relation failed")
    return tuple(reasons)


def build_candidate_report(
    *,
    candidate_checkpoint: Path,
    suite_path: Path,
    cases: Sequence[CandidateValidationCase],
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate separately persisted simulator results into one freeze decision."""
    case_rows = []
    total_episodes = 0
    successful_episodes = 0
    random_layout_episodes = 0
    for case in cases:
        case.validate()
        result = results.get(case.name)
        if result is None:
            reasons = ("result file is missing",)
            successes = 0
            result_path = None
        else:
            reasons = assess_candidate_result(result, expected_episodes=case.num_envs)
            successes = int(result.get("chain_successes", 0))
            result_path = result.get("_result_path")
        total_episodes += case.num_envs
        successful_episodes += max(0, min(successes, case.num_envs))
        if case.layout == "random_safe":
            random_layout_episodes += case.num_envs
        case_rows.append({
            "name": case.name,
            "seed": case.seed,
            "num_envs": case.num_envs,
            "layout": case.layout,
            "layout_seed": case.layout_seed,
            "passed": not reasons,
            "rejection_reasons": list(reasons),
            "chain_successes": successes,
            "result": result_path,
        })
    passed_cases = sum(int(row["passed"]) for row in case_rows)
    passed = passed_cases == len(cases) and successful_episodes == total_episodes
    return {
        "schema": "stage_vla_v5.action_candidate_report.v1",
        "status": "passed" if passed else "failed",
        "freeze_recommended": passed,
        "candidate_checkpoint": str(Path(candidate_checkpoint).resolve()),
        "suite": str(Path(suite_path).resolve()),
        "cases_passed": passed_cases,
        "cases_total": len(cases),
        "episodes_passed": successful_episodes,
        "episodes_total": total_episodes,
        "random_layout_episodes": random_layout_episodes,
        "requirements": {
            "all_cases_pass": True,
            "all_episodes_pass": True,
            "pure_policy_actions": True,
            "maximum_mid_episode_resets": 0,
            "geometric_reach_recovery_allowed": False,
            "final_stability_speed_source": "control_delta",
        },
        "cases": case_rows,
    }
