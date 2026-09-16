"""Frozen v4.2 expert-dataset contract.

This module is intentionally Isaac-free so the critical label and leakage
checks can be unit-tested with normal Python.

The v4.1 audit found a concrete failure in the legacy phase mapper: substring
matching for ``"close"`` mapped ``"R8 closed-loop horizontal refinement"`` to
GRASP.  v4.2 therefore uses an allow-list plus full-match regular expressions;
there are no substring fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import re
from typing import Any, Mapping, Sequence


PILOT_EPISODE_SCHEMA = "stage_vla_v4.v4_2_episode.v1"
PILOT_DATASET_SCHEMA = "stage_vla_v4.v4_2_pilot_dataset.v1"
REPLAY_REPORT_SCHEMA = "stage_vla_v4.v4_2_replay_report.v1"
SOURCE_TRACE_SCHEMA = "stage_vla.m11.m7_continuous_trace.v1"

TRAIN_SEED_MIN = 2000
TRAIN_SEED_MAX = 2999
FINAL_HELD_OUT_SEEDS = frozenset(range(1100, 1120))
LEGACY_REGRESSION_SEEDS = frozenset((1004, 1009, 1015, 1020, 1031))


class V4Stage(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT = 2
    TRANSPORT = 3
    PLACE = 4


STAGE_NAMES = tuple(stage.name for stage in V4Stage)

# Exact phase labels observed in the accepted R8 expert.  The two placement
# attempt phases are handled below with full-match regexes because the attempt
# number may be 1..N.
_EXACT_PHASE_TO_STAGE: dict[str, V4Stage] = {
    "approach / adaptive descend": V4Stage.REACH,
    "close / stable grasp": V4Stage.GRASP,
    "M6 lift reproduction": V4Stage.LIFT,
    "pre-transport hold": V4Stage.TRANSPORT,
    "vertical clearance": V4Stage.TRANSPORT,
    "R8 object-space Y waypoint": V4Stage.TRANSPORT,
    "R8 object-space X waypoint": V4Stage.TRANSPORT,
    # IMPORTANT: this contains the text "closed-loop" but is TRANSPORT.
    "R8 closed-loop horizontal refinement": V4Stage.TRANSPORT,
    "release": V4Stage.PLACE,
    "open-gripper retract": V4Stage.PLACE,
    "settle / M7 success gates": V4Stage.PLACE,
}

_PLACEMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"R7 pre-place XY align attempt [1-9][0-9]*\Z"),
    re.compile(r"R7 Z-only placement descent attempt [1-9][0-9]*\Z"),
)


# v4.2 state vector.  Every position is ENV-LOCAL, not world-frame, to prevent
# the multi-environment world-coordinate leakage diagnosed in V3 D9.
STATE_FIELD_NAMES: tuple[str, ...] = (
    *(f"robot_joint_pos[{i}]" for i in range(9)),
    *(f"robot_joint_vel[{i}]" for i in range(9)),
    "ee_pos_local[x]", "ee_pos_local[y]", "ee_pos_local[z]",
    "left_tip_local[x]", "left_tip_local[y]", "left_tip_local[z]",
    "right_tip_local[x]", "right_tip_local[y]", "right_tip_local[z]",
    "red_pos_local[x]", "red_pos_local[y]", "red_pos_local[z]",
    "red_quat_wxyz[w]", "red_quat_wxyz[x]", "red_quat_wxyz[y]", "red_quat_wxyz[z]",
    "red_lin_vel_w[x]", "red_lin_vel_w[y]", "red_lin_vel_w[z]",
    "red_ang_vel_w[x]", "red_ang_vel_w[y]", "red_ang_vel_w[z]",
    "blue_pos_local[x]", "blue_pos_local[y]", "blue_pos_local[z]",
    "blue_quat_wxyz[w]", "blue_quat_wxyz[x]", "blue_quat_wxyz[y]", "blue_quat_wxyz[z]",
    "blue_lin_vel_w[x]", "blue_lin_vel_w[y]", "blue_lin_vel_w[z]",
    "blue_ang_vel_w[x]", "blue_ang_vel_w[y]", "blue_ang_vel_w[z]",
    "gripper_joint_pos[left]", "gripper_joint_pos[right]",
    "finger_force_n[left]", "finger_force_n[right]",
)
STATE_DIM = len(STATE_FIELD_NAMES)
assert STATE_DIM == 57


@dataclass(frozen=True)
class TraceValidation:
    seed: int
    num_steps: int
    stage_ids: tuple[int, ...]
    stage_names: tuple[str, ...]
    phase_names: tuple[str, ...]


def map_expert_phase(phase: str) -> V4Stage:
    """Map one accepted expert phase to the frozen five-stage vocabulary.

    This is deliberately strict.  Unknown phases abort collection rather than
    being guessed from words such as ``close``, ``lift`` or ``place``.
    """
    if not isinstance(phase, str) or not phase:
        raise ValueError("expert phase must be a non-empty string")
    if phase in _EXACT_PHASE_TO_STAGE:
        return _EXACT_PHASE_TO_STAGE[phase]
    if any(pattern.fullmatch(phase) for pattern in _PLACEMENT_PATTERNS):
        return V4Stage.PLACE
    raise ValueError(
        f"unknown expert phase {phase!r}; v4.2 forbids substring/fuzzy stage mapping"
    )


def validate_train_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed in FINAL_HELD_OUT_SEEDS:
        raise ValueError(f"seed {seed} is FINAL held-out and forbidden for v4.2 data")
    if seed in LEGACY_REGRESSION_SEEDS:
        raise ValueError(f"seed {seed} is legacy-regression-only and forbidden for v4.2 data")
    if not TRAIN_SEED_MIN <= seed <= TRAIN_SEED_MAX:
        raise ValueError(
            f"seed {seed} is outside frozen v4 TRAIN range {TRAIN_SEED_MIN}..{TRAIN_SEED_MAX}"
        )


def validate_stage_sequence(stage_ids: Sequence[int], *, require_all_stages: bool = True) -> None:
    if not stage_ids:
        raise ValueError("stage sequence is empty")
    ints = [int(value) for value in stage_ids]
    if any(value < int(V4Stage.REACH) or value > int(V4Stage.PLACE) for value in ints):
        raise ValueError(f"stage sequence contains invalid ids: {sorted(set(ints))}")
    regressions = [(i - 1, ints[i - 1], ints[i]) for i in range(1, len(ints)) if ints[i] < ints[i - 1]]
    if regressions:
        i, prev, cur = regressions[0]
        raise ValueError(f"stage sequence regressed at {i}->{i + 1}: {prev}->{cur}")
    if require_all_stages:
        missing = [stage.name for stage in V4Stage if int(stage) not in set(ints)]
        if missing:
            raise ValueError(f"trace is missing required stages: {missing}")


def _validate_action(action: Any, *, step: int) -> None:
    if not isinstance(action, list) or len(action) != 7:
        raise ValueError(f"trace step {step}: raw_action must contain exactly 7 values")
    values = [float(value) for value in action]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"trace step {step}: raw_action contains NaN/Inf")
    if values[6] not in (-1.0, 1.0):
        raise ValueError(f"trace step {step}: gripper must be -1 or +1, got {values[6]}")


def validate_trace_payload_for_v4_2(payload: Mapping[str, Any]) -> TraceValidation:
    """Validate one raw state-machine expert trace for v4.2 admission."""
    if payload.get("schema") != SOURCE_TRACE_SCHEMA:
        raise ValueError(f"unsupported source trace schema: {payload.get('schema')!r}")
    if payload.get("strict_success") is not True:
        raise ValueError("source trace is not declared strict_success=true")
    seed = int(payload.get("seed"))
    validate_train_seed(seed)
    warmup = payload.get("warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("trace warmup_steps must be a non-negative integer")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source trace records must be a non-empty list")

    stage_ids: list[int] = []
    phase_names: list[str] = []
    for step, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"trace record {step} must be a mapping")
        phase = record.get("phase")
        stage = map_expert_phase(phase)
        _validate_action(record.get("raw_action"), step=step)
        stage_ids.append(int(stage))
        phase_names.append(str(phase))

    validate_stage_sequence(stage_ids, require_all_stages=True)

    # Phase/action causal sanity.  REACH stays open; once GRASP begins the expert
    # remains closed through TRANSPORT.  PLACE may be closed during align/descent
    # and becomes open only for release/retract/settle.
    for step, (record, stage_id) in enumerate(zip(records, stage_ids, strict=True)):
        grip = float(record["raw_action"][6])
        if stage_id == int(V4Stage.REACH) and grip != 1.0:
            raise ValueError(f"trace step {step}: REACH must keep gripper open")
        if stage_id in (int(V4Stage.GRASP), int(V4Stage.LIFT), int(V4Stage.TRANSPORT)) and grip != -1.0:
            raise ValueError(f"trace step {step}: GRASP/LIFT/TRANSPORT must keep gripper closed")

    return TraceValidation(
        seed=seed,
        num_steps=len(records),
        stage_ids=tuple(stage_ids),
        stage_names=tuple(V4Stage(value).name for value in stage_ids),
        phase_names=tuple(phase_names),
    )
