"""Strict audit for generated Isaac Lab SkillGen cube-stack datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_TARGET_ENV = "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"
REQUIRED_OBSERVATIONS = ("object", "eef_pos", "eef_quat", "gripper_pos")


def sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_generated_skillgen_dataset(
    path: str | Path, *, minimum_demonstrations: int = 1
) -> dict:
    """Validate generated training data without requiring seed-only annotations."""
    if not isinstance(minimum_demonstrations, int) or isinstance(minimum_demonstrations, bool):
        raise ValueError("minimum_demonstrations must be an integer")
    if minimum_demonstrations < 1:
        raise ValueError("minimum_demonstrations must be positive")
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    episode_rows = []
    action_blocks = []
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError("generated SkillGen dataset has no /data group")
        data = dataset["data"]
        if "env_args" not in data.attrs:
            raise ValueError("generated SkillGen dataset has no env_args metadata")
        env_args = json.loads(data.attrs["env_args"])
        if env_args.get("env_name") != EXPECTED_TARGET_ENV:
            raise ValueError(f"unexpected generated environment: {env_args}")

        names = sorted(data, key=lambda value: int(value.rsplit("_", 1)[1]))
        if len(names) < minimum_demonstrations:
            raise ValueError(
                f"expected at least {minimum_demonstrations} demonstrations, found {len(names)}"
            )
        for name in names:
            episode = data[name]
            actions = np.asarray(episode["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise ValueError(f"{name}: expected finite [T,7] actions")
            samples = int(episode.attrs["num_samples"])
            if samples != len(actions) or samples < 2:
                raise ValueError(f"{name}: invalid num_samples")
            if not bool(episode.attrs.get("success", False)):
                raise ValueError(f"{name}: generated training set contains a failed episode")
            if "obs" not in episode:
                raise ValueError(f"{name}: missing observations")
            for key in REQUIRED_OBSERVATIONS:
                if key not in episode["obs"] or len(episode[f"obs/{key}"]) != samples:
                    raise ValueError(f"{name}: observation {key} length mismatch")
            action_blocks.append(actions)
            episode_rows.append({"name": name, "samples": samples, "success": True})

    actions = np.concatenate(action_blocks)
    return {
        "version": "skillgen-franka-cube-stack-generated-audit-v1",
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "environment": EXPECTED_TARGET_ENV,
        "demonstrations": len(episode_rows),
        "successful_demonstrations": len(episode_rows),
        "transitions": int(len(actions)),
        "action_dim": 7,
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "action_mean_abs": np.abs(actions).mean(axis=0).tolist(),
        "episodes": episode_rows,
    }
