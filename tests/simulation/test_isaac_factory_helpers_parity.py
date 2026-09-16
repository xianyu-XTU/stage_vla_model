from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from stage_vla_v7.simulation.isaac_lab.contact_sensors import (
    install_finger_net_contact_sensors,
)
from stage_vla_v7.simulation.isaac_lab.fixed_object_pose import (
    set_fixed_asset_poses,
    set_fixed_pair_pose,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.envs.contact_sensors import (  # noqa: E402
    install_finger_net_contact_sensors as v5_install_finger_net_contact_sensors,
)
from stage_vla.envs.fixed_object_pose import (  # noqa: E402
    set_fixed_asset_poses as v5_set_fixed_asset_poses,
    set_fixed_pair_pose as v5_set_fixed_pair_pose,
)


def _sensor_config():
    spawn = SimpleNamespace(activate_contact_sensors=False)
    return SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)))


def test_fingertip_contact_sensor_config_matches_v5() -> None:
    actual = _sensor_config()
    expected = _sensor_config()
    kwargs = dict(update_period=0.02, history_length=7, debug_vis=True)
    install_finger_net_contact_sensors(actual, **kwargs)
    v5_install_finger_net_contact_sensors(expected, **kwargs)
    assert actual.scene.robot.spawn.activate_contact_sensors
    assert expected.scene.robot.spawn.activate_contact_sensors
    for name in ("left_finger_contact", "right_finger_contact"):
        actual_sensor = getattr(actual.scene, name)
        expected_sensor = getattr(expected.scene, name)
        for field in ("prim_path", "update_period", "history_length", "debug_vis"):
            assert getattr(actual_sensor, field) == getattr(expected_sensor, field)


class FakeAsset:
    def __init__(self, quaternion: torch.Tensor) -> None:
        self.data = SimpleNamespace(root_quat_w=quaternion.clone())
        self.pose = None
        self.pose_ids = None
        self.velocity = None
        self.velocity_ids = None

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids) -> None:
        self.pose = root_pose.clone()
        self.pose_ids = env_ids.clone()

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids) -> None:
        self.velocity = root_velocity.clone()
        self.velocity_ids = env_ids.clone()


class FakeScene(dict):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.env_origins = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
        quaternion = torch.randn(count, 4, generator=torch.Generator().manual_seed(19))
        for name in ("cube_1", "cube_2", "cube_3"):
            self[name] = FakeAsset(quaternion)


def _environment(count: int = 4):
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=count,
        scene=FakeScene(count),
    )


def _assert_writes_match(actual, expected, names) -> None:
    for name in names:
        left = actual.scene[name]
        right = expected.scene[name]
        assert torch.equal(left.pose, right.pose)
        assert torch.equal(left.pose_ids, right.pose_ids)
        assert torch.equal(left.velocity, right.velocity)
        assert torch.equal(left.velocity_ids, right.velocity_ids)
        assert torch.count_nonzero(left.velocity) == 0


def test_fixed_pair_pose_matches_v5_exactly() -> None:
    actual = _environment()
    expected = _environment()
    ids = torch.tensor([0, 2])
    kwargs = dict(blue_xyz=(0.45, 0.10, 0.02), red_xyz=(0.40, -0.08, 0.02))
    set_fixed_pair_pose(actual, ids, **kwargs)
    v5_set_fixed_pair_pose(expected, ids, **kwargs)
    _assert_writes_match(actual, expected, ("cube_1", "cube_2"))


def test_fixed_asset_poses_match_v5_exactly() -> None:
    actual = _environment()
    expected = _environment()
    ids = torch.tensor([1, 3])
    positions = {
        "cube_1": ((0.45, 0.10, 0.02),) * 4,
        "cube_2": ((0.40, -0.08, 0.02),) * 4,
        "cube_3": ((0.55, -0.14, 0.02),) * 4,
    }
    set_fixed_asset_poses(actual, ids, asset_xyz=positions)
    v5_set_fixed_asset_poses(expected, ids, asset_xyz=positions)
    _assert_writes_match(actual, expected, tuple(positions))
