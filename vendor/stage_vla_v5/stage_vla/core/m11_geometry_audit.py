"""Pure helpers for the M11-B-D9 expert reference-replay audit.

The D9 expert audit never applies the counterfactual D9 GRIP token to the
environment.  The environment executes the saved expert token at every step.
Consequently, replay error after the D9 CLOSE decision is useful diagnostic
evidence, but it cannot have been caused by the unexecuted D9 CLOSE action.

This module keeps that distinction explicit and dependency-free so the causal
prefix contract can be regression-tested without launching Isaac Lab.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


M11_D9_EXPERT_AUDIT_SCHEMA = "stage_vla.m11.geometry_ready_expert_audit.v2"
M11_D9_EXPERT_AUDIT_SUMMARY_SCHEMA = (
    "stage_vla.m11.geometry_ready_expert_audit_summary.v2"
)
M11_D9_REFERENCE_ACTION_SOURCE = "saved_expert_token_indices"


def _normalise_error_rows(
    observation_abs_errors: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not observation_abs_errors:
        raise ValueError("observation_abs_errors must contain at least one step")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for step, raw_row in enumerate(observation_abs_errors):
        row = tuple(float(value) for value in raw_row)
        if not row:
            raise ValueError(f"observation error row {step} is empty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("observation error rows must have equal width")
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ValueError("observation absolute errors must be finite and >= 0")
        rows.append(row)
    return tuple(rows)


def _peak(
    rows: tuple[tuple[float, ...], ...],
    *,
    start_step: int,
    end_step_exclusive: int,
    tolerance: float,
) -> dict:
    if start_step >= end_step_exclusive:
        return {
            "available": False,
            "start_step_inclusive": start_step,
            "end_step_inclusive": None,
            "maximum_abs_error": None,
            "maximum_error_step": None,
            "maximum_error_dimension": None,
            "reproduced_within_tolerance": None,
        }
    maximum = -1.0
    maximum_step: int | None = None
    maximum_dimension: int | None = None
    for step in range(start_step, end_step_exclusive):
        for dimension, value in enumerate(rows[step]):
            if value > maximum:
                maximum = value
                maximum_step = step
                maximum_dimension = dimension
    return {
        "available": True,
        "start_step_inclusive": start_step,
        "end_step_inclusive": end_step_exclusive - 1,
        "maximum_abs_error": maximum,
        "maximum_error_step": maximum_step,
        "maximum_error_dimension": maximum_dimension,
        "reproduced_within_tolerance": maximum <= tolerance,
    }


def _first_exceedance(
    rows: tuple[tuple[float, ...], ...], tolerance: float
) -> dict | None:
    for step, row in enumerate(rows):
        maximum = max(row)
        if maximum > tolerance:
            return {
                "step": step,
                "dimension": row.index(maximum),
                "maximum_abs_error": maximum,
            }
    return None


def build_reference_replay_diagnostics(
    observation_abs_errors: Sequence[Sequence[float]],
    *,
    d9_close_decision_step: int | None,
    tolerance: float,
) -> dict:
    """Split expert replay fidelity into causal-prefix and later diagnostics.

    The causal prefix includes the observation used to choose D9 CLOSE at
    ``d9_close_decision_step``.  The next observation belongs to the
    post-decision region.  D9 itself remains counterfactual throughout both
    regions; every environment step must use the saved expert action.
    """

    rows = _normalise_error_rows(observation_abs_errors)
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and > 0")
    tolerance = float(tolerance)
    if d9_close_decision_step is not None and not (
        0 <= int(d9_close_decision_step) < len(rows)
    ):
        raise ValueError("d9_close_decision_step is outside the replay")
    decision_step = (
        int(d9_close_decision_step)
        if d9_close_decision_step is not None
        else None
    )
    causal_end_exclusive = (
        decision_step + 1 if decision_step is not None else len(rows)
    )
    causal_prefix = _peak(
        rows,
        start_step=0,
        end_step_exclusive=causal_end_exclusive,
        tolerance=tolerance,
    )
    post_decision = _peak(
        rows,
        start_step=causal_end_exclusive,
        end_step_exclusive=len(rows),
        tolerance=tolerance,
    )
    full_trajectory = _peak(
        rows,
        start_step=0,
        end_step_exclusive=len(rows),
        tolerance=tolerance,
    )
    initial = _peak(
        rows,
        start_step=0,
        end_step_exclusive=1,
        tolerance=tolerance,
    )
    first = _first_exceedance(rows, tolerance)
    if first is None:
        divergence_phase = "none"
    elif first["step"] == 0:
        divergence_phase = "initial"
    elif decision_step is None:
        divergence_phase = "undetermined_without_d9_close"
    elif first["step"] <= decision_step:
        divergence_phase = "causal_prefix"
    else:
        divergence_phase = "post_d9_close_decision"

    return {
        "observation_error_tolerance": tolerance,
        "executed_action_source": M11_D9_REFERENCE_ACTION_SOURCE,
        "d9_grip_applied_to_environment": False,
        "d9_close_decision_step": decision_step,
        "causal_prefix_definition": (
            "steps 0..predicted_d9_close inclusive, before executing that step"
        ),
        "full_trajectory_is_diagnostic_only": True,
        "initial_observation": initial,
        "causal_prefix": causal_prefix,
        "post_decision": post_decision,
        "full_trajectory": full_trajectory,
        "first_tolerance_exceedance": first,
        "divergence_phase": divergence_phase,
        "causal_prefix_authorized": (
            decision_step is not None
            and causal_prefix["reproduced_within_tolerance"] is True
        ),
    }
