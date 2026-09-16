from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from stage_vla_v7.simulation.isaac_lab.snapshot_state import expand_single_env_state


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.place_snapshot import expand_single_env_state as v5_expand_single_env_state  # noqa: E402


def _assert_nested_parity(actual, expected) -> None:
    assert type(actual) is type(expected)
    if torch.is_tensor(actual):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_parity(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_parity(left, right)
    else:
        assert actual == expected


def test_expand_single_env_state_matches_v5_exactly() -> None:
    state = {
        "articulation": {
            "robot": {
                "joint_position": torch.arange(7, dtype=torch.float32).reshape(1, 7),
                "joint_velocity": torch.arange(7, dtype=torch.float64).reshape(1, 7),
                "scalar": torch.tensor(3, dtype=torch.int64),
            }
        },
        "rigid_object": [
            torch.tensor([[1.0, 2.0, 3.0]]),
            (torch.tensor([[True, False]]), {"name": "cube_2"}),
        ],
    }
    actual = expand_single_env_state(state, 5, "cpu")
    expected = v5_expand_single_env_state(state, 5, "cpu")
    _assert_nested_parity(actual, expected)
    assert actual["articulation"]["robot"]["joint_position"].shape == (5, 7)
    actual["articulation"]["robot"]["joint_position"][0, 0] = -99.0
    assert state["articulation"]["robot"]["joint_position"][0, 0] == 0.0


@pytest.mark.parametrize("count", [0, -1])
def test_expand_single_env_state_rejects_invalid_count_like_v5(count: int) -> None:
    state = {"value": torch.ones((1, 2))}
    with pytest.raises(ValueError, match="n must be > 0"):
        expand_single_env_state(state, count, "cpu")
    with pytest.raises(ValueError, match="n must be > 0"):
        v5_expand_single_env_state(state, count, "cpu")


def test_expand_single_env_state_rejects_non_singleton_batch_like_v5() -> None:
    state = {"value": torch.ones((2, 3))}
    with pytest.raises(ValueError, match="batch dimension 1"):
        expand_single_env_state(state, 2, "cpu")
    with pytest.raises(ValueError, match="batch dimension 1"):
        v5_expand_single_env_state(state, 2, "cpu")
