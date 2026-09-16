from __future__ import annotations

import math

import pytest

from stage_vla_v7.simulation import (
    CubeModel,
    DepthCameraModel,
    FrankaModel,
    RGBCameraModel,
    StackScene,
    default_environment_registry,
    default_model_registry,
    default_scene_registry,
)


def test_default_simulation_model_registries_are_typed_and_fail_closed() -> None:
    registry = default_model_registry()

    assert isinstance(registry.require_robot("franka"), FrankaModel)
    assert isinstance(registry.require_object("cube"), CubeModel)
    assert isinstance(registry.require_sensor("rgb"), RGBCameraModel)
    assert isinstance(registry.require_sensor("depth"), DepthCameraModel)
    with pytest.raises(LookupError, match="unknown robot"):
        registry.require_robot("ur5")
    with pytest.raises(LookupError, match="unknown object"):
        registry.require_object("cylinder")
    with pytest.raises(LookupError, match="unknown sensor"):
        registry.require_sensor("lidar")


def test_scene_and_environment_registries_fail_closed() -> None:
    scene = default_scene_registry().require("stack")
    environment = default_environment_registry().require("red_on_blue")

    assert isinstance(scene, StackScene)
    assert environment.scene.load().robot_id == "franka-panda"
    with pytest.raises(LookupError, match="unknown scene"):
        default_scene_registry().require("warehouse")
    with pytest.raises(LookupError, match="unknown environment"):
        default_environment_registry().require("pick_only")
    with pytest.raises(TypeError, match="SimulationScene"):
        default_scene_registry().register("invalid", object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SimulationEnvironment"):
        default_environment_registry().register("invalid", object())  # type: ignore[arg-type]


def test_stack_scene_loads_required_entities_and_resets_deterministically() -> None:
    scene = StackScene()
    manifest = scene.load()
    first = scene.reset(61081)
    second = scene.reset(61081)
    different = scene.reset(61082)

    assert manifest.includes_table and manifest.includes_light
    assert manifest.robot_id == "franka-panda"
    assert len(manifest.object_ids) == 3
    assert len(manifest.sensor_ids) == 2
    assert first == second
    assert first.object_positions_xyz_m != different.object_positions_xyz_m
    positions = tuple(first.object_positions_xyz_m.values())
    assert all(position[2] == pytest.approx(0.0203) for position in positions)
    assert all(
        math.dist(left[:2], right[:2]) >= scene.minimum_separation_m
        for index, left in enumerate(positions)
        for right in positions[index + 1 :]
    )


def test_invalid_simulation_model_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        CubeModel(size_m=0.0)
    with pytest.raises(ValueError):
        RGBCameraModel(width=0)
    with pytest.raises(ValueError):
        DepthCameraModel(minimum_depth_m=2.0, maximum_depth_m=1.0)
    with pytest.raises(TypeError, match="SimulationModelDescriptor"):
        default_model_registry().register_robot("invalid", object())
    with pytest.raises(ValueError, match="cannot accept model type"):
        default_model_registry().register_robot("cube-as-robot", CubeModel())
