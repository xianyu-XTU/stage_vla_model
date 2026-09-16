"""Episode-safe action chunk construction for v4 expert demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_action_chunks(actions: np.ndarray, starts: np.ndarray, chunk_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build zero-padded chunks without ever crossing an episode boundary."""
    source = np.asarray(actions, dtype=np.float32)
    indices = np.asarray(starts, dtype=np.int64).reshape(-1)
    if source.ndim != 2 or source.shape[1] != 7 or not np.isfinite(source).all():
        raise ValueError("actions must be finite [T,7]")
    if chunk_length not in (10, 20):
        raise ValueError("chunk_length must be 10 or 20")
    if np.any(indices < 0) or np.any(indices >= len(source)):
        raise ValueError("chunk start is outside the source episode")
    chunks = np.zeros((len(indices), chunk_length, 7), dtype=np.float32)
    mask = np.zeros((len(indices), chunk_length), dtype=np.bool_)
    for row, start in enumerate(indices.tolist()):
        count = min(chunk_length, len(source) - start)
        chunks[row, :count] = source[start : start + count]
        mask[row, :count] = True
    return chunks, mask


def load_episode_chunk_samples(
    episode_path: str | Path,
    *,
    chunk_length: int = 20,
    stride: int = 5,
    max_samples: int | None = None,
) -> dict[str, np.ndarray | str | int]:
    """Load one accepted episode and select causal pre-action chunk starts."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    path = Path(episode_path)
    with np.load(path, allow_pickle=False) as episode:
        required = {"images", "states", "actions", "step_id", "episode_id", "seed", "instruction"}
        missing = required.difference(episode.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        images = np.asarray(episode["images"], dtype=np.uint8)
        states = np.asarray(episode["states"], dtype=np.float32)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        step_ids = np.asarray(episode["step_id"], dtype=np.int64)
        episode_ids = np.asarray(episode["episode_id"], dtype=np.int64)
        seeds = np.asarray(episode["seed"], dtype=np.int64)
        instruction = str(np.asarray(episode["instruction"]).item())
    length = len(actions)
    if images.shape[0] != length or states.shape != (length, 57) or actions.shape != (length, 7):
        raise ValueError(f"{path.name}: incompatible image/state/action lengths")
    if not np.array_equal(step_ids, np.arange(length)):
        raise ValueError(f"{path.name}: non-contiguous step ids")
    if len(np.unique(episode_ids)) != 1 or len(np.unique(seeds)) != 1:
        raise ValueError(f"{path.name}: episode or seed changes inside trajectory")
    if not 2000 <= int(seeds[0]) <= 2999:
        raise ValueError(f"{path.name}: only frozen TRAIN seeds may enter prototype training")
    starts = np.arange(0, length, stride, dtype=np.int64)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive when specified")
        starts = starts[:max_samples]
    chunks, mask = build_action_chunks(actions, starts, chunk_length)
    return {
        "images": images[starts],
        "states": states[starts],
        "action_chunks": chunks,
        "action_mask": mask,
        "source_step_id": step_ids[starts],
        "episode_id": int(episode_ids[0]),
        "seed": int(seeds[0]),
        "instruction": instruction,
    }
