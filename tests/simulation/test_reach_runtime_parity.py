from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from stage_vla_v7.simulation.environments.reach_observation import (
    REACH_OBS_DIM,
    reach_observation,
)
from stage_vla_v7.simulation.isaac_lab.action_adapter import reach_raw_action
from stage_vla_v7.simulation.isaac_lab.reach_state import measure_reach_state


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.reach_policy import REACH_OBS_DIM as V5_REACH_OBS_DIM  # noqa: E402
from stage_vla.rl.reach_policy import measure_reach_state as v5_measure_reach_state  # noqa: E402
from stage_vla.rl.reach_policy import reach_observation as v5_reach_observation  # noqa: E402
from stage_vla.rl.reach_policy import reach_raw_action as v5_reach_raw_action  # noqa: E402


class _Proxy:
    def __init__(self, value: torch.Tensor) -> None:
        self.torch = value


class _Robot:
    def __init__(self, position: torch.Tensor, velocity: torch.Tensor) -> None:
        self.data = SimpleNamespace(joint_pos=_Proxy(position), joint_vel=_Proxy(velocity))

    @staticmethod
    def find_joints(names, preserve_order=False):
        if names == ["panda_joint[1-7]"]:
            assert preserve_order
            return list(range(7)), None
        return [7, 8], None


def _fake_environment(seed: int = 61081, count: int = 5):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    def asset():
        return SimpleNamespace(data=SimpleNamespace(
            root_pos_w=_Proxy(rand(count, 3)),
            root_lin_vel_w=_Proxy(rand(count, 3)),
            root_ang_vel_w=_Proxy(rand(count, 3)),
            root_quat_w=_Proxy(rand(count, 4)),
        ))

    frame_names = ["tool_rightfinger", "end_effector", "tool_leftfinger"]
    frame = SimpleNamespace(data=SimpleNamespace(
        target_frame_names=frame_names,
        target_pos_w=_Proxy(rand(count, len(frame_names), 3)),
    ))
    left = SimpleNamespace(data=SimpleNamespace(net_forces_w=_Proxy(rand(count, 2, 3))))
    right = SimpleNamespace(data=SimpleNamespace(net_forces_w=_Proxy(rand(count, 1, 3))))
    joint_position = rand(count, 9)
    joint_position[:, 7:] = 0.04
    return SimpleNamespace(
        scene={
            "cube_1": asset(),
            "cube_2": asset(),
            "cube_3": asset(),
            "ee_frame": frame,
            "left_finger_contact": left,
            "right_finger_contact": right,
            "robot": _Robot(joint_position, rand(count, 9)),
        },
        cfg=SimpleNamespace(
            gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
            gripper_open_val=0.04,
        ),
    )


def _assert_tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_measure_reach_state_matches_v5_exactly() -> None:
    env = _fake_environment()
    actual = measure_reach_state(
        env, object_asset_name="cube_3", support_asset_name="cube_1"
    )
    expected = v5_measure_reach_state(
        env, object_asset_name="cube_3", support_asset_name="cube_1"
    )
    assert actual.keys() == expected.keys()
    for key in actual:
        _assert_tensor_parity(actual[key], expected[key])


@pytest.mark.parametrize("use_grasp_target", [False, True])
def test_reach_observation_matches_v5_exactly(use_grasp_target: bool) -> None:
    generator = torch.Generator().manual_seed(47)
    count = 11

    def rand(width: int) -> torch.Tensor:
        return torch.randn((count, width), generator=generator)

    state = {
        "ee": rand(3),
        "red": rand(3),
        "blue": rand(3),
        "left_tip": rand(3),
        "right_tip": rand(3),
        "speed": torch.rand(count, generator=generator),
        "red_vel": rand(3),
        "red_ang": rand(3),
        "open": torch.rand(count, generator=generator) > 0.5,
        "grip": rand(2),
        "q": rand(7),
        "qd": rand(7),
        "red_quat": rand(4),
        "blue_quat": rand(4),
    }
    if use_grasp_target:
        state["grasp_target"] = rand(3)
    previous = rand(5)
    actual = reach_observation(state, previous)
    expected = v5_reach_observation(state, previous)
    assert REACH_OBS_DIM == V5_REACH_OBS_DIM == 52
    _assert_tensor_parity(actual, expected)
    assert float((actual - expected).abs().max()) == 0.0


def test_reach_raw_action_matches_v5_exactly() -> None:
    action = torch.tensor(
        [[-2.0, -0.5, 0.25, 1.5, -1.0], [0.1, 0.2, 0.3, 0.4, 0.5]],
        dtype=torch.float64,
    )
    actual = reach_raw_action(action)
    expected = v5_reach_raw_action(action)
    _assert_tensor_parity(actual, expected)
    assert actual.dtype == torch.float32
    assert torch.count_nonzero(actual[:, 3:5]) == 0
    assert torch.equal(actual[:, 6], torch.ones(2))


def test_reach_helpers_reject_invalid_shapes_like_v5() -> None:
    with pytest.raises(ValueError, match=r"\[N,5\]"):
        reach_raw_action(torch.zeros(5))
    state = {"ee": torch.zeros((2, 3)), "red": torch.zeros((2, 3))}
    with pytest.raises(ValueError, match=r"\[N,5\]"):
        reach_observation(state, torch.zeros((2, 4)))
