from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from stage_vla_v7.simulation.physics import ACTION_CONTEXT_VERSION
from tools.evaluation.data_collection import SkillDemonstrationBuffer


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.skill_demonstrations import SkillDemonstrationBuffer as V5SkillDemonstrationBuffer  # noqa: E402


def _assert_payload_parity(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        if torch.is_tensor(actual[key]):
            assert actual[key].shape == expected[key].shape
            assert actual[key].dtype == expected[key].dtype
            assert torch.equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


def test_demonstration_buffer_matches_v5_exactly() -> None:
    actual = SkillDemonstrationBuffer(("reach", "grasp"))
    expected = V5SkillDemonstrationBuffer(("reach", "grasp"))
    generator = torch.Generator().manual_seed(61081)
    for skill, width, count in (("REACH", 52, 3), ("GRASP", 55, 4)):
        observations = torch.randn((count, width), generator=generator, dtype=torch.float64)
        actions = 3.0 * torch.randn((count, 5), generator=generator, dtype=torch.float64)
        actual.append(skill, observations, actions)
        expected.append(skill, observations, actions)
    assert actual.sample_counts() == expected.sample_counts() == {"REACH": 3, "GRASP": 4}
    for skill in ("REACH", "GRASP"):
        _assert_payload_parity(
            actual.payload(skill, source="parity"),
            expected.payload(skill, source="parity"),
        )


def test_physical_context_payload_contract_matches_v5() -> None:
    actual = SkillDemonstrationBuffer(("GRASP",))
    expected = V5SkillDemonstrationBuffer(("GRASP",))
    observations = torch.arange(130, dtype=torch.float32).reshape(2, 65)
    actions = torch.zeros((2, 5))
    actual.append("GRASP", observations, actions)
    expected.append("GRASP", observations, actions)
    metadata = {
        "state_observation_dim": 55,
        "object_context_version": ACTION_CONTEXT_VERSION,
        "object_geometry": "box",
    }
    _assert_payload_parity(
        actual.payload("GRASP", **metadata),
        expected.payload("GRASP", **metadata),
    )


@pytest.mark.parametrize(
    ("observations", "actions", "message"),
    [
        (torch.zeros(5), torch.zeros((1, 5)), r"\[N,D\]"),
        (torch.zeros((1, 55)), torch.zeros((1, 4)), r"\[N,5\]"),
        (torch.full((1, 55), float("nan")), torch.zeros((1, 5)), "non-finite"),
    ],
)
def test_demonstration_buffer_validation_matches_v5(
    observations: torch.Tensor,
    actions: torch.Tensor,
    message: str,
) -> None:
    for buffer in (SkillDemonstrationBuffer(("GRASP",)), V5SkillDemonstrationBuffer(("GRASP",))):
        with pytest.raises(ValueError, match=message):
            buffer.append("GRASP", observations, actions)


def test_demonstration_payload_rejects_reserved_metadata_like_v5() -> None:
    for buffer in (SkillDemonstrationBuffer(("GRASP",)), V5SkillDemonstrationBuffer(("GRASP",))):
        buffer.append("GRASP", torch.zeros((1, 55)), torch.zeros((1, 5)))
        with pytest.raises(ValueError, match="reserved fields"):
            buffer.payload("GRASP", action_dim=7)
