from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from stage_vla_v7.simulation.config import (
    RigidObjectMotionProfile,
    motion_profile_from_config,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.stages.object_motion import RigidObjectMotionProfile as V5RigidObjectMotionProfile  # noqa: E402
from stage_vla.stages.object_motion import motion_profile_from_config as v5_motion_profile_from_config  # noqa: E402


@pytest.mark.parametrize(
    "motion",
    [
        {},
        {"lift_translation_limit_m": 0.004},
        {
            "lift_translation_limit_m": 0.001,
            "transport_translation_limit_m": 0.002,
            "align_translation_limit_m": 0.003,
            "descend_translation_limit_m": 0.004,
            "release_translation_limit_m": 0.005,
            "retreat_translation_limit_m": 0.006,
        },
    ],
)
def test_motion_profile_loader_matches_v5(tmp_path: Path, motion: dict[str, float]) -> None:
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"motion": motion}), encoding="utf-8")
    actual = motion_profile_from_config(path)
    expected = v5_motion_profile_from_config(path)
    assert asdict(actual) == asdict(expected)
    for skill in (
        "LIFT",
        "TRANSPORT",
        "ALIGN",
        "DESCEND",
        "RELEASE_STABILIZE",
        "RETREAT",
    ):
        assert actual.for_skill(skill) == expected.for_skill(skill)


def test_motion_profile_defaults_and_validation_match_v5() -> None:
    assert asdict(RigidObjectMotionProfile()) == asdict(V5RigidObjectMotionProfile())
    for value in (0.0, -0.001, 0.011):
        actual = RigidObjectMotionProfile(lift_translation_limit_m=value)
        expected = V5RigidObjectMotionProfile(lift_translation_limit_m=value)
        with pytest.raises(ValueError, match=r"\(0, 0.01\]"):
            actual.validate()
        with pytest.raises(ValueError, match=r"\(0, 0.01\]"):
            expected.validate()
    with pytest.raises(ValueError, match="no motion limit"):
        RigidObjectMotionProfile().for_skill("GRASP")
