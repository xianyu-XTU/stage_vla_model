from __future__ import annotations

import copy
import math

import pytest

from tools.generalization.manifest import (
    ASSETS,
    GROUND_Z_M,
    MINIMUM_SEPARATION_M,
    generate_layout_manifest,
    load_phase4_manifest,
    slice_manifest,
    validate_manifest_payload,
    write_manifest,
)


def test_phase4_manifest_is_deterministic_and_legal(tmp_path) -> None:
    first = generate_layout_manifest(seed=42024, layout_count=20)
    second = generate_layout_manifest(seed=42024, layout_count=20)
    assert first == second
    assert first["layout_count"] == 20
    assert len(first["layout_ids"]) == 20

    positions = first["asset_positions_local_xyz"]
    for index in range(20):
        points = [positions[asset][index] for asset in ASSETS]
        assert all(point[2] == GROUND_Z_M for point in points)
        assert min(
            math.dist(points[left][:2], points[right][:2])
            for left in range(3)
            for right in range(left + 1, 3)
        ) >= MINIMUM_SEPARATION_M

    path = write_manifest(tmp_path / "layouts.json", first)
    assert load_phase4_manifest(path) == first


def test_manifest_slice_preserves_layout_identity() -> None:
    manifest = generate_layout_manifest(seed=42024, layout_count=20)
    batch = slice_manifest(manifest, (4, 5, 6, 7))
    assert batch["layout_ids"] == [
        "layout_005",
        "layout_006",
        "layout_007",
        "layout_008",
    ]
    assert batch["layout_count"] == 4
    for asset in ASSETS:
        assert batch["asset_positions_local_xyz"][asset] == (
            manifest["asset_positions_local_xyz"][asset][4:8]
        )


def test_manifest_rejects_wrong_ground_height() -> None:
    manifest = generate_layout_manifest(seed=42024, layout_count=2)
    invalid = copy.deepcopy(manifest)
    invalid["asset_positions_local_xyz"]["cube_2"][0][2] = 0.0
    with pytest.raises(ValueError, match="ground height"):
        validate_manifest_payload(invalid)
