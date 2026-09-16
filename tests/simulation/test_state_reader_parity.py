from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from stage_vla_v7.simulation.isaac_lab.state_reader import (
    net_force_per_env,
    read_grasp_state,
    read_placement_state,
    to_torch,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.envs.state_readers import (  # noqa: E402
    net_force_per_env as v5_net_force_per_env,
    read_grasp_state as v5_read_grasp_state,
    read_placement_state as v5_read_placement_state,
)


class Proxy:
    def __init__(self, value: torch.Tensor) -> None:
        self.torch = value


class Robot:
    def __init__(self, joint_position: torch.Tensor) -> None:
        self.data = SimpleNamespace(joint_pos=Proxy(joint_position))

    @staticmethod
    def find_joints(_names):
        return [7, 8], None


def _fake_environment(seed: int = 19, count: int = 4):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    def asset() -> object:
        return SimpleNamespace(data=SimpleNamespace(
            root_pos_w=Proxy(rand(count, 3)),
            root_lin_vel_w=Proxy(rand(count, 3)),
            root_ang_vel_w=Proxy(rand(count, 3)),
        ))

    frame_names = ["tool_rightfinger", "end_effector", "tool_leftfinger"]
    frame = SimpleNamespace(data=SimpleNamespace(
        target_frame_names=frame_names,
        target_pos_w=Proxy(rand(count, len(frame_names), 3)),
    ))
    left_force = SimpleNamespace(data=SimpleNamespace(net_forces_w=Proxy(rand(count, 2, 3))))
    right_force = SimpleNamespace(data=SimpleNamespace(net_forces_w=Proxy(rand(count, 1, 3))))
    joints = rand(count, 9)
    joints[:, 7:9] = 0.04
    return SimpleNamespace(
        scene={
            "cube_2": asset(),
            "cube_1": asset(),
            "ee_frame": frame,
            "left_finger_contact": left_force,
            "right_finger_contact": right_force,
            "robot": Robot(joints),
        },
        cfg=SimpleNamespace(
            gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
            gripper_open_val=0.04,
        ),
    )


def test_to_torch_handles_isaac_proxy_array() -> None:
    value = torch.arange(6).reshape(2, 3)
    assert to_torch(Proxy(value)) is value


def test_net_force_matches_v5() -> None:
    env = _fake_environment()
    sensor = env.scene["left_finger_contact"]
    actual = net_force_per_env(sensor)
    expected = v5_net_force_per_env(sensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_grasp_state_matches_v5() -> None:
    env = _fake_environment()
    actual = read_grasp_state(env)
    expected = v5_read_grasp_state(env)
    pairs = (
        (actual.object_position_w, expected.red_pos_w),
        (actual.end_effector_position_w, expected.ee_pos_w),
        (actual.left_fingertip_position_w, expected.left_tip_w),
        (actual.right_fingertip_position_w, expected.right_tip_w),
        (actual.finger_a_force_n, expected.finger_a_force_n),
        (actual.finger_b_force_n, expected.finger_b_force_n),
    )
    assert all(left.shape == right.shape for left, right in pairs)
    assert all(left.dtype == right.dtype for left, right in pairs)
    assert all(torch.equal(left, right) for left, right in pairs)


def test_placement_state_matches_v5() -> None:
    env = _fake_environment()
    actual = read_placement_state(env)
    expected = v5_read_placement_state(env)
    pairs = (
        (actual.object_position_w, expected.red_pos_w),
        (actual.support_position_w, expected.blue_pos_w),
        (actual.object_linear_velocity_w, expected.red_lin_vel_w),
        (actual.object_angular_velocity_w, expected.red_ang_vel_w),
        (actual.support_linear_velocity_w, expected.blue_lin_vel_w),
        (actual.support_angular_velocity_w, expected.blue_ang_vel_w),
        (actual.gripper_joint_position, expected.gripper_joint_pos),
        (actual.gripper_open, expected.gripper_open),
    )
    assert all(left.shape == right.shape for left, right in pairs)
    assert all(left.dtype == right.dtype for left, right in pairs)
    assert all(torch.equal(left, right) for left, right in pairs)
