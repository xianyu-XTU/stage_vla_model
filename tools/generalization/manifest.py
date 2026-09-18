"""Generate and validate frozen Phase 4 layout manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from tools.evaluation.bootstrap import ensure_v7_source_available

ensure_v7_source_available()

from stage_vla_v7.simulation.randomization import (
    load_layout_manifest,
    sample_red_blue_batch,
    seeded_random,
)


SCHEMA = "stage_vla_v7.asset_layout_batch.v1"
MANIFEST_VERSION = "phase4_layouts_v1"
GENERATOR = "tools.generalization.manifest.generate_layout_manifest"
ASSETS = ("cube_1", "cube_2", "cube_3")
OBJECT_ASSET = "cube_2"
SUPPORT_ASSET = "cube_1"
DISTRACTOR_ASSET = "cube_3"
X_RANGE_M = (0.4, 0.6)
Y_RANGE_M = (-0.1, 0.1)
GROUND_Z_M = 0.0203
MINIMUM_SEPARATION_M = 0.1


def _sample_distractor(
    *,
    seed: int,
    index: int,
    occupied: Sequence[Sequence[float]],
) -> list[float]:
    randomizer = seeded_random(int(seed) + 1_000_003 + int(index))
    for _attempt in range(10_000):
        candidate = [
            randomizer.uniform(*X_RANGE_M),
            randomizer.uniform(*Y_RANGE_M),
            GROUND_Z_M,
        ]
        if all(
            math.dist(candidate[:2], point[:2]) >= MINIMUM_SEPARATION_M
            for point in occupied
        ):
            return candidate
    raise RuntimeError(f"could not sample a legal distractor for layout {index}")


def generate_layout_manifest(*, seed: int, layout_count: int = 20) -> dict[str, object]:
    """Generate a deterministic legal batch from the evaluator's pair sampler."""
    count = int(layout_count)
    if count < 1:
        raise ValueError("layout_count must be positive")
    pairs = sample_red_blue_batch(
        count,
        seed=int(seed),
        minimum_separation_m=MINIMUM_SEPARATION_M,
    )
    distractors = [
        _sample_distractor(
            seed=int(seed),
            index=index,
            occupied=(pairs["blue"][index], pairs["red"][index]),
        )
        for index in range(count)
    ]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "generator": GENERATOR,
        "seed": int(seed),
        "layout_count": count,
        "num_envs": count,
        "layout_ids": [f"layout_{index + 1:03d}" for index in range(count)],
        "object_asset": OBJECT_ASSET,
        "support_asset": SUPPORT_ASSET,
        "distractor_asset": DISTRACTOR_ASSET,
        "domain": {
            "x_range_m": list(X_RANGE_M),
            "y_range_m": list(Y_RANGE_M),
            "ground_z_m": GROUND_Z_M,
            "workspace_boundary": "inclusive",
        },
        "constraints": {
            "minimum_xy_separation_m": MINIMUM_SEPARATION_M,
            "all_assets_in_workspace": True,
            "finite_xyz": True,
            "fixed_ground_z": True,
        },
        "minimum_separation_m": MINIMUM_SEPARATION_M,
        "asset_positions_local_xyz": {
            SUPPORT_ASSET: pairs["blue"],
            OBJECT_ASSET: pairs["red"],
            DISTRACTOR_ASSET: distractors,
        },
    }
    validate_manifest_payload(payload)
    return payload


def validate_manifest_payload(payload: Mapping[str, object]) -> None:
    """Reject non-finite, out-of-domain, overlapping, or mis-sized layouts."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported manifest schema: {payload.get('schema')!r}")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unexpected phase4 manifest version")
    count = int(payload.get("layout_count", 0))
    if count < 1 or int(payload.get("num_envs", 0)) != count:
        raise ValueError("layout_count and num_envs must be equal and positive")
    layout_ids = payload.get("layout_ids")
    if not isinstance(layout_ids, list) or len(layout_ids) != count:
        raise ValueError("layout_ids must contain one identifier per layout")
    if len(set(str(value) for value in layout_ids)) != count:
        raise ValueError("layout_ids must be unique")
    positions = payload.get("asset_positions_local_xyz")
    if not isinstance(positions, Mapping) or set(positions) != set(ASSETS):
        raise ValueError(f"manifest assets must be exactly {list(ASSETS)}")
    for asset in ASSETS:
        rows = positions[asset]
        if not isinstance(rows, Sequence) or len(rows) != count:
            raise ValueError(f"{asset} must contain {count} XYZ rows")
        for row in rows:
            if not isinstance(row, Sequence) or len(row) != 3:
                raise ValueError(f"{asset} positions must be XYZ triples")
            xyz = tuple(float(value) for value in row)
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"{asset} positions must be finite")
            if not X_RANGE_M[0] <= xyz[0] <= X_RANGE_M[1]:
                raise ValueError(f"{asset} X is outside the frozen workspace")
            if not Y_RANGE_M[0] <= xyz[1] <= Y_RANGE_M[1]:
                raise ValueError(f"{asset} Y is outside the frozen workspace")
            if not math.isclose(xyz[2], GROUND_Z_M, abs_tol=1e-12):
                raise ValueError(f"{asset} Z must equal the frozen cube ground height")
    for index in range(count):
        points = [positions[asset][index] for asset in ASSETS]
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                if (
                    math.dist(points[left][:2], points[right][:2]) + 1e-12
                    < MINIMUM_SEPARATION_M
                ):
                    raise ValueError(
                        f"layout {layout_ids[index]} violates minimum XY separation"
                    )


def load_phase4_manifest(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_manifest_payload(payload)
    load_layout_manifest(
        resolved,
        required_assets=ASSETS,
        expected_num_envs=int(payload["layout_count"]),
    )
    return payload


def slice_manifest(
    payload: Mapping[str, object],
    indices: Sequence[int],
) -> dict[str, object]:
    validate_manifest_payload(payload)
    selected = tuple(int(index) for index in indices)
    count = int(payload["layout_count"])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("manifest slice indices must be non-empty and unique")
    if min(selected) < 0 or max(selected) >= count:
        raise IndexError("manifest slice index is out of range")
    positions = payload["asset_positions_local_xyz"]
    assert isinstance(positions, Mapping)
    result = dict(payload)
    result["parent_manifest_version"] = payload["manifest_version"]
    result["parent_layout_count"] = count
    result["layout_count"] = len(selected)
    result["num_envs"] = len(selected)
    result["layout_ids"] = [payload["layout_ids"][index] for index in selected]
    result["asset_positions_local_xyz"] = {
        asset: [positions[asset][index] for index in selected]
        for asset in ASSETS
    }
    validate_manifest_payload(result)
    return result


def write_manifest(path: Path, payload: Mapping[str, object]) -> Path:
    validate_manifest_payload(payload)
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "ASSETS",
    "DISTRACTOR_ASSET",
    "GENERATOR",
    "GROUND_Z_M",
    "MANIFEST_VERSION",
    "MINIMUM_SEPARATION_M",
    "OBJECT_ASSET",
    "SCHEMA",
    "SUPPORT_ASSET",
    "X_RANGE_M",
    "Y_RANGE_M",
    "generate_layout_manifest",
    "load_phase4_manifest",
    "slice_manifest",
    "validate_manifest_payload",
    "write_manifest",
]
