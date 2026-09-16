"""Serializable evaluation report helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .metrics import SuccessMetrics


def evaluation_report(
    metrics: Mapping[str, SuccessMetrics],
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": "stage_vla_v7.action_evaluation.v1",
        "skills": {
            name: {
                "successes": value.successes,
                "trials": value.trials,
                "rate": value.rate,
                "confidence_95": value.confidence_95,
            }
            for name, value in sorted(metrics.items())
        },
        "metadata": dict(metadata or {}),
    }
