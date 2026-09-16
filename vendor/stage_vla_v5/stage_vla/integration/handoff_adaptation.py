"""TRAIN-only live-handoff dataset contracts for M19 transition adaptation.

M19-D1/D2 used the historical five train seeds.  M19-D3 keeps those protocols
backward-compatible but adds a *disjoint* diverse-training seed range so the
Recovery Adapter can see many independent TRANSPORT->PLACE transitions without
using the fixed held-out evaluation set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

TRAIN_SEEDS = (1004, 1009, 1015, 1020, 1031)
HELD_OUT_SEEDS = tuple(range(1100, 1120))
D3_TRAIN_SEED_MIN = 2000
D3_TRAIN_SEED_MAX = 2999
D3_MIN_SNAPSHOTS = 50
D3_MAX_SNAPSHOTS = 100
D3_DEFAULT_SNAPSHOTS = 64
HANDOFF_ROLE = "live_handoff"


def validate_train_seed(seed: int) -> int:
    """Historical D1/D2 contract: exactly the five fixed train seeds."""
    seed = int(seed)
    if seed not in TRAIN_SEEDS:
        raise ValueError(
            f"live-handoff adaptation forbids held-out/test seed {seed}; "
            f"allowed TRAIN seeds={TRAIN_SEEDS}"
        )
    return seed


def validate_d3_train_seed(seed: int) -> int:
    """D3 contract: a disjoint TRAIN-only seed range, never held-out seeds."""
    seed = int(seed)
    if seed in HELD_OUT_SEEDS:
        raise ValueError(f"M19-D3 forbids held-out seed {seed} in training")
    if not D3_TRAIN_SEED_MIN <= seed <= D3_TRAIN_SEED_MAX:
        raise ValueError(
            f"M19-D3 TRAIN seed {seed} outside reserved range "
            f"[{D3_TRAIN_SEED_MIN}, {D3_TRAIN_SEED_MAX}]"
        )
    return seed


def validate_d3_snapshot_seed_set(
    seeds: Iterable[int],
    *,
    min_count: int = D3_MIN_SNAPSHOTS,
    max_count: int = D3_MAX_SNAPSHOTS,
) -> tuple[int, ...]:
    """Validate a diverse D3 snapshot bank before it can enter PPO training."""
    values = tuple(int(s) for s in seeds)
    if not min_count <= len(values) <= max_count:
        raise ValueError(
            f"M19-D3 requires {min_count}..{max_count} snapshots; got {len(values)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("M19-D3 requires one independent snapshot per unique source seed")
    for seed in values:
        validate_d3_train_seed(seed)
    return values


def discover_handoff_snapshots(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("seed_*_handoff.pt"))


def sample_handoff_mask(n: int, fraction: float, generator: torch.Generator) -> torch.Tensor:
    """Return a CPU bool mask selecting live-handoff resets."""
    if n <= 0:
        return torch.zeros((0,), dtype=torch.bool)
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("handoff_fraction must be in [0,1]")
    if fraction <= 0.0:
        return torch.zeros((n,), dtype=torch.bool)
    if fraction >= 1.0:
        return torch.ones((n,), dtype=torch.bool)
    return torch.rand((n,), generator=generator) < fraction
