"""Pure file/manifest helpers for v4.2 pilot data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .v4_2_contract import (
    PILOT_EPISODE_SCHEMA,
    STATE_DIM,
    STATE_FIELD_NAMES,
    V4Stage,
    validate_stage_sequence,
    validate_train_seed,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_spans(stage_ids: np.ndarray) -> dict[str, list[int]]:
    ids = np.asarray(stage_ids, dtype=np.int64).reshape(-1)
    validate_stage_sequence(ids.tolist(), require_all_stages=True)
    out: dict[str, list[int]] = {}
    for stage in V4Stage:
        indices = np.flatnonzero(ids == int(stage))
        if indices.size:
            out[stage.name] = [int(indices[0]), int(indices[-1]) + 1]
    return out


def validate_episode_npz(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "images", "states", "actions", "stage_id", "stage_name", "source_phase",
            "episode_id", "step_id", "seed", "reset_reason", "terminal_reason",
            "state_field_names", "instruction", "metadata_json",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path.name}: missing arrays {missing}")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if metadata.get("schema") != PILOT_EPISODE_SCHEMA:
            raise ValueError(f"{path.name}: episode schema mismatch")

        images = np.asarray(data["images"])
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        stage_ids = np.asarray(data["stage_id"], dtype=np.int64).reshape(-1)
        step_ids = np.asarray(data["step_id"], dtype=np.int64).reshape(-1)
        episode_ids = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
        seeds = np.asarray(data["seed"], dtype=np.int64).reshape(-1)
        reset_reason = np.asarray(data["reset_reason"]).astype(str).reshape(-1)
        terminal_reason = np.asarray(data["terminal_reason"]).astype(str).reshape(-1)
        n = len(actions)

        if images.ndim != 4 or images.shape[0] != n or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"{path.name}: images must be uint8 [T,H,W,3], got {images.shape}/{images.dtype}")
        if states.shape != (n, STATE_DIM):
            raise ValueError(f"{path.name}: states must be [{n},{STATE_DIM}], got {states.shape}")
        if actions.shape != (n, 7) or not np.isfinite(actions).all():
            raise ValueError(f"{path.name}: actions must be finite [{n},7]")
        if not np.all(np.isin(actions[:, 6], (-1.0, 1.0))):
            raise ValueError(f"{path.name}: gripper action is not binary -1/+1")
        if not np.isfinite(states).all():
            raise ValueError(f"{path.name}: state contains NaN/Inf")
        if not np.array_equal(step_ids, np.arange(n, dtype=np.int64)):
            raise ValueError(f"{path.name}: step_id is not contiguous 0..T-1")
        if len(set(episode_ids.tolist())) != 1:
            raise ValueError(f"{path.name}: episode_id changed inside trajectory")
        if len(set(seeds.tolist())) != 1:
            raise ValueError(f"{path.name}: seed changed inside trajectory")
        validate_train_seed(int(seeds[0]))
        validate_stage_sequence(stage_ids.tolist(), require_all_stages=True)
        if np.any(reset_reason != ""):
            raise ValueError(f"{path.name}: post-reset sample detected")
        if n < 1 or terminal_reason[-1] != "strict_success" or np.any(terminal_reason[:-1] != ""):
            raise ValueError(f"{path.name}: terminal_reason contract violated")
        fields = tuple(np.asarray(data["state_field_names"]).astype(str).tolist())
        if fields != STATE_FIELD_NAMES:
            raise ValueError(f"{path.name}: state field schema mismatch")

        expected_names = np.asarray([V4Stage(int(v)).name for v in stage_ids])
        if not np.array_equal(np.asarray(data["stage_name"]).astype(str), expected_names):
            raise ValueError(f"{path.name}: stage_name does not match stage_id")

        return {
            "path": str(path),
            "seed": int(seeds[0]),
            "episode_id": int(episode_ids[0]),
            "steps": n,
            "image_shape": list(images.shape[1:]),
            "stage_spans": stage_spans(stage_ids),
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
        }


def find_trace_jsons(trace_root: str | Path) -> list[Path]:
    root = Path(trace_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(set(int(value) for value in values))
