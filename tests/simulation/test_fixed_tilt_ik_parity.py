from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from stage_vla_v7.simulation.physics.fixed_tilt import (
    FixedTiltReference,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    summarize_tilt_trace,
    wrap_angle,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.fixed_tilt_reference import FixedTiltReference as V5FixedTiltReference  # noqa: E402
from stage_vla.rl.fixed_tilt_reference import euler_xyz_from_quat as v5_euler_xyz_from_quat  # noqa: E402
from stage_vla.rl.fixed_tilt_reference import quat_from_euler_xyz as v5_quat_from_euler_xyz  # noqa: E402
from stage_vla.rl.fixed_tilt_reference import summarize_tilt_trace as v5_summarize_tilt_trace  # noqa: E402
from stage_vla.rl.fixed_tilt_reference import wrap_angle as v5_wrap_angle  # noqa: E402


def _assert_tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_fixed_tilt_math_and_reference_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(61081)
    rpy = torch.randn((8, 3), generator=generator)
    actual_quaternion = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    expected_quaternion = v5_quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    _assert_tensor_parity(actual_quaternion, expected_quaternion)
    for actual, expected in zip(
        euler_xyz_from_quat(actual_quaternion),
        v5_euler_xyz_from_quat(expected_quaternion),
    ):
        _assert_tensor_parity(actual, expected)
    angles = torch.linspace(-12.0, 12.0, 101)
    _assert_tensor_parity(wrap_angle(angles), v5_wrap_angle(angles))

    actual = FixedTiltReference(8, "cpu")
    expected = V5FixedTiltReference(8, "cpu")
    for _ in range(4):
        delta = 0.1 * torch.randn(8, generator=generator)
        _assert_tensor_parity(
            actual.target(actual_quaternion, delta),
            expected.target(expected_quaternion, delta),
        )
        _assert_tensor_parity(actual.reference_rpy, expected.reference_rpy)
        _assert_tensor_parity(actual.desired_yaw, expected.desired_yaw)
        assert torch.equal(actual.initialized, expected.initialized)
    reset_ids = torch.tensor([1, 5, 7])
    actual.reset(reset_ids)
    expected.reset(reset_ids)
    assert torch.equal(actual.initialized, expected.initialized)


def test_fixed_tilt_trace_summary_matches_v5() -> None:
    count = 3
    rpy = torch.tensor([[0.2, -0.3, 0.4], [-0.1, 0.1, -0.5], [0.3, 0.2, 0.0]])
    quaternion = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    trace = {
        "frames": {
            "ee_quat_xyzw": torch.stack((quaternion, quaternion)),
            "ik_desired_quat_xyzw": torch.stack((quaternion, quaternion)),
            "tilt_reference_rpy": torch.stack((rpy, rpy)),
            "tilt_initialized": torch.ones((2, count), dtype=torch.bool),
        }
    }
    assert summarize_tilt_trace(trace) == v5_summarize_tilt_trace(trace)


class _Controller:
    def __init__(self, count: int) -> None:
        self.ee_pos_des = torch.zeros((count, 3))
        self.ee_quat_des = torch.zeros((count, 4))

    def set_command(self, actions, ee_pos, ee_quat) -> None:
        self.ee_pos_des[:] = ee_pos + actions[:, :3]
        self.ee_quat_des[:] = ee_quat


class _FakeDifferentialIK:
    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = torch.device("cpu")
        self.action_dim = 6
        self._raw_actions = torch.zeros((self.num_envs, 6))
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.as_tensor(env.scale, dtype=torch.float32)
        self._clip = torch.as_tensor(env.clip, dtype=torch.float32)
        self._ik_controller = _Controller(self.num_envs)
        self._position = env.position.clone()
        self._quaternion = env.quaternion.clone()
        self.reset_ids = None

    @property
    def raw_actions(self):
        return self._raw_actions

    def _compute_frame_pose(self):
        return self._position, self._quaternion

    def reset(self, env_ids=None) -> None:
        self.reset_ids = env_ids


def _install_isaac_action_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (
        "isaaclab",
        "isaaclab.envs",
        "isaaclab.envs.mdp",
        "isaaclab.envs.mdp.actions",
        "isaaclab.envs.mdp.actions.task_space_actions",
    )
    modules = {name: ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules[names[-1]].DifferentialInverseKinematicsAction = _FakeDifferentialIK
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_fixed_tilt_isaac_action_matches_v5(monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.import_module("stage_vla_v7.simulation.isaac_lab")
    _install_isaac_action_stub(monkeypatch)
    native_module = importlib.import_module(
        "stage_vla_v7.simulation.isaac_lab.fixed_tilt_ik"
    )
    legacy_module = importlib.import_module("stage_vla.envs.fixed_tilt_ik")
    cfg = SimpleNamespace(
        controller=SimpleNamespace(use_relative_mode=True, command_type="pose"),
        clip=True,
    )
    rpy = torch.tensor([[0.2, -0.3, 0.4], [-0.1, 0.1, -0.5]])
    env = SimpleNamespace(
        num_envs=2,
        scale=torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.25]),
        clip=torch.tensor([[[-0.2, 0.2]] * 6, [[-0.2, 0.2]] * 6]),
        position=torch.tensor([[0.4, 0.0, 0.2], [0.5, 0.1, 0.3]]),
        quaternion=quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2]),
    )
    actual = native_module.FixedTiltDifferentialInverseKinematicsAction(cfg, env)
    expected = legacy_module.FixedTiltDifferentialInverseKinematicsAction(cfg, env)
    actions = torch.tensor(
        [[0.3, -0.1, 0.05, 0.0, 0.0, 0.4], [-0.4, 0.2, -0.1, 0.0, 0.0, -0.5]]
    )
    actual.process_actions(actions)
    expected.process_actions(actions)
    _assert_tensor_parity(actual._raw_actions, expected._raw_actions)
    _assert_tensor_parity(actual._processed_actions, expected._processed_actions)
    for left, right in zip(actual.desired_pose, expected.desired_pose):
        _assert_tensor_parity(left, right)
    with pytest.raises(ValueError, match="roll/pitch"):
        actual.process_actions(torch.ones((2, 6)))
