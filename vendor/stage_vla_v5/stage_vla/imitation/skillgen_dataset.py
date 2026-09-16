"""Validation and indexing for the official Franka SkillGen cube-stack data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_ENV = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
SUBTASKS = ("grasp_1", "stack_1", "grasp_2", "stack_2")


@dataclass(frozen=True)
class EpisodeIndex:
    name: str
    samples: int
    success: bool
    start: dict[str, int]
    termination: dict[str, int | None]


def _first_true(values: h5py.Dataset) -> int:
    indices = np.flatnonzero(np.asarray(values, dtype=bool).reshape(-1))
    if not len(indices):
        raise ValueError(f"signal {values.name} never becomes true")
    return int(indices[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_skillgen_dataset(path: str | Path) -> dict:
    """Validate the upstream HDF5 contract and return a serializable audit."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes: list[EpisodeIndex] = []
    action_blocks = []
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError("SkillGen dataset has no /data group")
        data = dataset["data"]
        env_args = json.loads(data.attrs["env_args"])
        if env_args.get("env_name") != EXPECTED_ENV:
            raise ValueError(f"unexpected source environment: {env_args}")
        for name in sorted(data, key=lambda value: int(value.rsplit("_", 1)[1])):
            episode = data[name]
            actions = np.asarray(episode["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise ValueError(f"{name}: expected finite [T,7] actions")
            samples = int(episode.attrs["num_samples"])
            if samples != len(actions):
                raise ValueError(f"{name}: num_samples does not match actions")
            info = episode["obs/datagen_info"]
            start = {key: _first_true(info[f"subtask_start_signals/{key}"]) for key in SUBTASKS}
            termination = {
                key: _first_true(info[f"subtask_term_signals/{key}"]) for key in SUBTASKS[:-1]
            }
            termination["stack_2"] = None
            ordered = [start["grasp_1"], termination["grasp_1"], start["stack_1"],
                       termination["stack_1"], start["grasp_2"], termination["grasp_2"],
                       start["stack_2"], samples]
            if ordered != sorted(ordered) or ordered[0] < 0 or ordered[-1] != samples:
                raise ValueError(f"{name}: invalid SkillGen boundary order {ordered}")
            for key in ("eef_pos", "eef_quat", "gripper_pos", "object"):
                if len(episode[f"obs/{key}"]) != samples:
                    raise ValueError(f"{name}: observation {key} length mismatch")
            episodes.append(EpisodeIndex(name=name, samples=samples,
                                         success=bool(episode.attrs.get("success", False)),
                                         start=start, termination=termination))
            action_blocks.append(actions)
    if not episodes:
        raise ValueError("SkillGen dataset contains no demonstrations")
    actions = np.concatenate(action_blocks)
    return {
        "version": "skillgen-franka-cube-stack-audit-v1",
        "dataset": str(path),
        "dataset_sha256": _sha256(path),
        "source_environment": EXPECTED_ENV,
        "target_environment": "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
        "demonstrations": len(episodes),
        "successful_demonstrations": sum(item.success for item in episodes),
        "transitions": int(len(actions)),
        "action_dim": int(actions.shape[1]),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "action_mean_abs": np.abs(actions).mean(axis=0).tolist(),
        "subtask_order": list(SUBTASKS),
        "episodes": [asdict(item) for item in episodes],
    }
