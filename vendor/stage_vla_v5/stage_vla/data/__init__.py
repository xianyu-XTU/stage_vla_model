"""Dataset contracts and utilities for stage_vla_v4."""

from .v4_2_contract import (
    PILOT_DATASET_SCHEMA,
    PILOT_EPISODE_SCHEMA,
    REPLAY_REPORT_SCHEMA,
    STATE_FIELD_NAMES,
    STATE_DIM,
    TRAIN_SEED_MAX,
    TRAIN_SEED_MIN,
    V4Stage,
    map_expert_phase,
    validate_stage_sequence,
    validate_trace_payload_for_v4_2,
)

__all__ = [
    "PILOT_DATASET_SCHEMA",
    "PILOT_EPISODE_SCHEMA",
    "REPLAY_REPORT_SCHEMA",
    "STATE_FIELD_NAMES",
    "STATE_DIM",
    "TRAIN_SEED_MAX",
    "TRAIN_SEED_MIN",
    "V4Stage",
    "map_expert_phase",
    "validate_stage_sequence",
    "validate_trace_payload_for_v4_2",
]
