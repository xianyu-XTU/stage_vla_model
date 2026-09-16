from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stage_vla_v7.simulation.config import CameraSpec, validate_camera_specs
from stage_vla_v7.simulation.isaac_lab import (
    OBSERVER_CAMERA_NAME,
    VISION_CAMERA_NAME,
    IsaacCameraAdapter,
    SimulationEnvironmentFactory,
)
from stage_vla_v7.simulation.models import ObserverCameraModel
from stage_vla_v7.simulation.scenes import StackScene
from tools.evaluation.camera_setup import build_evaluation_camera_specs


def _camera(value: int) -> object:
    rgb = np.full((1, 8, 8, 4), value, dtype=np.uint8)
    return SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb}))


def test_vision_and_observer_camera_ids_are_distinct() -> None:
    specs = build_evaluation_camera_specs(
        use_vision=True,
        observer=ObserverCameraModel(width=64, height=48),
        video_width=64,
        video_height=48,
    )
    assert {spec.role for spec in specs} == {"vision", "observer"}
    assert {spec.name for spec in specs} == {
        VISION_CAMERA_NAME,
        OBSERVER_CAMERA_NAME,
    }
    observer = next(spec for spec in specs if spec.role == "observer")
    assert observer.update_period_s == pytest.approx(1.0 / 20.0)


def test_observer_camera_fps_sets_sensor_update_period() -> None:
    specs = build_evaluation_camera_specs(
        use_vision=False,
        observer=ObserverCameraModel(width=64, height=48),
        video_width=64,
        video_height=48,
        video_fps=25.0,
    )
    assert specs[0].update_period_s == pytest.approx(1.0 / 25.0)


def test_duplicate_camera_id_fails_closed() -> None:
    vision = CameraSpec(
        VISION_CAMERA_NAME,
        "vision",
        8,
        8,
        ("rgb", "distance_to_image_plane"),
    )
    observer = CameraSpec(VISION_CAMERA_NAME, "observer", 8, 8, ("rgb",))
    with pytest.raises(ValueError, match="unique"):
        validate_camera_specs((vision, observer))


def test_vision_input_and_video_frame_use_different_cameras() -> None:
    vision = _camera(11)
    observer = _camera(222)
    adapter = IsaacCameraAdapter()
    bindings = adapter.bind(
        {VISION_CAMERA_NAME: vision, OBSERVER_CAMERA_NAME: observer},
        use_vision=True,
        record_video=True,
    )
    assert bindings.vision is vision
    assert bindings.observer is observer
    frame = adapter.observer_rgb_payload(observer, environment_index=0)
    assert np.asarray(frame)[0, 0, 0] == 222


def test_observer_camera_disabled_keeps_stack_scene_valid() -> None:
    manifest = StackScene(observer_camera=None).load()
    assert "stack-observer-camera" not in manifest.sensor_ids


def test_observer_camera_is_declared_only_when_enabled() -> None:
    manifest = StackScene(observer_camera=ObserverCameraModel()).load()
    assert "stack-observer-camera" in manifest.sensor_ids


def test_unknown_environment_fails_closed_without_launching_isaac() -> None:
    with pytest.raises(LookupError, match="unknown Simulation environment"):
        SimulationEnvironmentFactory().create("unknown")
