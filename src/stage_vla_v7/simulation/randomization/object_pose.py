"""Non-overlapping tabletop object-pose randomization."""

from __future__ import annotations

import math
from collections.abc import Sequence
import json
from pathlib import Path

from .seeds import seeded_random


def sample_object_positions(
    labels: Sequence[str],
    *,
    seed: int | None,
    x_range_m: tuple[float, float] = (0.40, 0.58),
    y_range_m: tuple[float, float] = (-0.15, 0.15),
    height_m: float = 0.0203,
    minimum_separation_m: float = 0.09,
    max_attempts_per_object: int = 1000,
) -> dict[str, tuple[float, float, float]]:
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("object labels must be non-empty and unique")
    if x_range_m[0] >= x_range_m[1] or y_range_m[0] >= y_range_m[1]:
        raise ValueError("object randomization ranges are invalid")
    if height_m <= 0.0 or minimum_separation_m <= 0.0:
        raise ValueError("object height and separation must be positive")
    randomizer = seeded_random(seed)
    result: dict[str, tuple[float, float, float]] = {}
    for label in labels:
        for _attempt in range(max_attempts_per_object):
            candidate = (
                randomizer.uniform(*x_range_m),
                randomizer.uniform(*y_range_m),
                height_m,
            )
            if all(
                math.hypot(candidate[0] - other[0], candidate[1] - other[1])
                >= minimum_separation_m
                for other in result.values()
            ):
                result[label] = candidate
                break
        else:
            raise RuntimeError("unable to sample a non-overlapping object layout")
    return result


def sample_red_blue_batch(
    num_envs: int,
    *,
    seed: int,
    minimum_separation_m: float = 0.1,
) -> dict[str, list[list[float]]]:
    """Reproduce the evaluator's frozen red/blue XY batch sampling contract."""
    if int(num_envs) < 1:
        raise ValueError("num_envs must be positive")
    randomizer = seeded_random(int(seed))
    blue_positions: list[list[float]] = []
    red_positions: list[list[float]] = []
    for _environment in range(int(num_envs)):
        for _attempt in range(10_000):
            blue = [
                randomizer.uniform(0.4, 0.6),
                randomizer.uniform(-0.1, 0.1),
                0.0203,
            ]
            red = [
                randomizer.uniform(0.4, 0.6),
                randomizer.uniform(-0.1, 0.1),
                0.0203,
            ]
            if math.dist(blue[:2], red[:2]) >= minimum_separation_m:
                blue_positions.append(blue)
                red_positions.append(red)
                break
        else:
            raise RuntimeError("could not sample valid red/blue XY positions")
    return {"blue": blue_positions, "red": red_positions}


def load_layout_manifest(
    path: Path,
    *,
    required_assets: Sequence[str],
    expected_num_envs: int,
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Load the frozen V5 layout schema through V7 Simulation ownership."""
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if payload.get("schema") not in {
        "stage_vla_v5.asset_layout_batch.v1",
        "stage_vla_v7.asset_layout_batch.v1",
    }:
        raise ValueError(f"unsupported asset layout schema: {payload.get('schema')!r}")
    raw_positions = payload.get("asset_positions_local_xyz")
    if not isinstance(raw_positions, dict):
        raise ValueError("asset layout must contain asset_positions_local_xyz")
    try:
        positions = {
            str(name): tuple(tuple(float(value) for value in row) for row in rows)
            for name, rows in raw_positions.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("asset layout positions must be arrays of XYZ rows") from exc
    required = tuple(required_assets)
    if set(positions) != set(required):
        raise ValueError(
            f"layout assets must be exactly {sorted(required)}, got {sorted(positions)}"
        )
    if int(expected_num_envs) < 1:
        raise ValueError("expected_num_envs must be positive")
    for asset, rows in positions.items():
        if len(rows) != int(expected_num_envs):
            raise ValueError(
                f"asset layout has {len(rows)} rows for {asset}, expected {expected_num_envs}"
            )
        for row in rows:
            if len(row) != 3 or not all(math.isfinite(value) for value in row):
                raise ValueError(f"{asset} positions must contain finite XYZ triples")
    separation = payload.get("minimum_separation_m")
    if separation is not None:
        separation = float(separation)
        if separation <= 0.0:
            raise ValueError("minimum separation must be positive")
        for env_index in range(int(expected_num_envs)):
            points = [positions[name][env_index] for name in required]
            if any(
                math.dist(points[left][:2], points[right][:2]) + 1e-12 < separation
                for left in range(len(points))
                for right in range(left + 1, len(points))
            ):
                raise ValueError(
                    f"environment {env_index} violates minimum XY separation"
                )
    return positions
