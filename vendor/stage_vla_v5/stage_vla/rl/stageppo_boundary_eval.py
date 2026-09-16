"""Aggregation helpers for frozen StagePPO boundary evaluations."""
from __future__ import annotations

import math


def wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not isinstance(successes, int) or not isinstance(attempts, int):
        raise ValueError("successes and attempts must be integers")
    if attempts < 1 or successes < 0 or successes > attempts:
        raise ValueError("invalid binomial counts")
    proportion = successes / attempts
    denominator = 1 + z * z / attempts
    center = (proportion + z * z / (2 * attempts)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / attempts + z * z / (4 * attempts * attempts)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate_boundary_evaluations(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one boundary evaluation is required")
    grouped: dict[float, list[dict]] = {}
    seen = set()
    for row in rows:
        position_range_m = float(row["position_range_m"])
        seed = int(row["eval_seed"])
        key = (position_range_m, seed)
        if key in seen:
            raise ValueError(f"duplicate evaluation {key}")
        seen.add(key)
        result = row["evaluation"]
        attempts, successes = int(result["attempts"]), int(result["successes"])
        if attempts < 1 or not 0 <= successes <= attempts:
            raise ValueError("invalid evaluation counts")
        if result.get("resets_during_rollout") != 0:
            raise ValueError("evaluation reset during rollout")
        grouped.setdefault(position_range_m, []).append(row)

    by_range = {}
    total_attempts = total_successes = 0
    for position_range_m, items in sorted(grouped.items()):
        attempts = sum(int(item["evaluation"]["attempts"]) for item in items)
        successes = sum(int(item["evaluation"]["successes"]) for item in items)
        failures = sum(int(item["evaluation"]["physical_failures"]) for item in items)
        timeouts = sum(int(item["evaluation"]["timeouts"]) for item in items)
        if successes + failures + timeouts != attempts:
            raise ValueError("terminal counts do not match attempts")
        low, high = wilson_interval(successes, attempts)
        by_range[f"{position_range_m:.6f}"] = {
            "position_range_m": position_range_m,
            "rng_seeds": sorted(int(item["eval_seed"]) for item in items),
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts,
            "wilson_95_interval": [low, high],
            "physical_failures": failures,
            "timeouts": timeouts,
        }
        total_attempts += attempts
        total_successes += successes
    low, high = wilson_interval(total_successes, total_attempts)
    return {
        "by_range": by_range,
        "overall": {
            "attempts": total_attempts,
            "successes": total_successes,
            "success_rate": total_successes / total_attempts,
            "wilson_95_interval": [low, high],
        },
    }


def aggregate_source_outcomes(evaluations: list[dict]) -> dict[str, dict]:
    rows = [row for evaluation in evaluations for row in evaluation["results"]]
    if not rows:
        raise ValueError("no per-source evaluation rows")
    result = {}
    for source_seed in sorted({int(row["source_seed"]) for row in rows}):
        selected = [row for row in rows if int(row["source_seed"]) == source_seed]
        attempts = len(selected)
        successes = sum(bool(row["success"]) for row in selected)
        failures = sum(bool(row["physical_failure"]) for row in selected)
        timeouts = sum(bool(row["timeout"]) for row in selected)
        if successes + failures + timeouts != attempts:
            raise ValueError("ambiguous source terminal accounting")
        low, high = wilson_interval(successes, attempts)
        result[str(source_seed)] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts,
            "wilson_95_interval": [low, high],
            "physical_failures": failures,
            "timeouts": timeouts,
        }
    return result


def recovery_acceptance(before, after, focus_source_seed, max_anchor_drop=2):
    """Require focus improvement, bounded anchor forgetting, and net improvement."""
    focus = str(focus_source_seed)
    if focus not in before or set(before) != set(after) or max_anchor_drop < 0:
        raise ValueError("invalid recovery acceptance inputs")
    before_focus = before[focus]["successes"]
    after_focus = after[focus]["successes"]
    anchors = [source for source in before if source != focus]
    before_anchor = sum(before[source]["successes"] for source in anchors)
    after_anchor = sum(after[source]["successes"] for source in anchors)
    before_total = before_focus + before_anchor
    after_total = after_focus + after_anchor
    accepted = (
        after_focus > before_focus
        and after_anchor >= before_anchor - max_anchor_drop
        and after_total > before_total
    )
    return {
        "accepted": accepted,
        "before_focus_successes": before_focus,
        "after_focus_successes": after_focus,
        "before_anchor_successes": before_anchor,
        "after_anchor_successes": after_anchor,
        "before_total_successes": before_total,
        "after_total_successes": after_total,
        "max_anchor_drop": max_anchor_drop,
    }
