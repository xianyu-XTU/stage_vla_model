from __future__ import annotations

import pytest

from stage_vla_v7.interfaces import (
    RobotAction,
    RobotObservation,
    SimulationAction,
    SimulationObservation,
    SimulationState,
)
from stage_vla_v7.simulation import RedOnBlueEnvironment
from stage_vla_v7.simulation.isaac_lab import (
    CallbackIsaacLabRuntime,
    IsaacActionAdapter,
    IsaacCameraAdapter,
    IsaacLabAdapter,
    IsaacObservationAdapter,
)


def _observation() -> SimulationObservation:
    return SimulationObservation(
        RobotObservation((0.1, 0.2, 0.3), "test-proprioception"),
        rgb=[[1, 2, 3]],
        depth_m=[[0.5]],
        frame_id="frame-7",
        timestamp_s=1.25,
        metadata={"environment_index": 0},
    )


def test_observation_camera_and_action_adapters_preserve_contracts() -> None:
    rows = IsaacObservationAdapter().batch([[1, 2, 3], [4, 5, 6]])
    frame = IsaacCameraAdapter().to_vision_request(
        [[1, 2, 3]],
        [[0.5]],
        frame_id="camera-1",
        metadata={"environment_index": 2},
    )
    action = IsaacActionAdapter().to_simulation_action(RobotAction(0.1, 0.2, 0.3, 0.4, 0.5))

    assert tuple(row.values for row in rows) == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert rows[1].metadata == {"batch_index": 1}
    assert frame.frame_id == "camera-1"
    assert frame.metadata == {"backend": "isaac-lab", "environment_index": 2}
    assert action.robot_action.values == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5))
    with pytest.raises(ValueError, match="rank 1"):
        IsaacObservationAdapter().to_robot_observation([[1.0, 2.0]])


def test_isaac_lab_adapter_implements_public_boundary() -> None:
    observation = _observation()
    adapter = IsaacLabAdapter()

    frame = adapter.to_vision_request(observation)
    assert frame.rgb is observation.rgb
    assert frame.depth_m is observation.depth_m
    assert adapter.to_robot_observation(observation) is observation.robot
    assert adapter.to_simulation_action(RobotAction(0, 0, 0, 0, 1)).control_mode == (
        "relative_cartesian_5d"
    )


def test_callback_runtime_drives_red_on_blue_environment() -> None:
    observation = _observation()
    calls: list[object] = []

    def reset(seed: int | None) -> SimulationState:
        calls.append(("reset", seed))
        return SimulationState("episode", 0, observation)

    def step(action: SimulationAction) -> SimulationState:
        calls.append(("step", action.robot_action.values))
        return SimulationState("episode", 1, observation, terminated=True)

    runtime = CallbackIsaacLabRuntime(
        reset=reset,
        observe=lambda: observation,
        step=step,
        close=lambda: calls.append("close"),
    )
    environment = RedOnBlueEnvironment(backend=runtime)
    state = environment.reset(61081)
    stepped = environment.step(
        SimulationAction(RobotAction(0.0, 0.0, 0.0, 0.0, 1.0))
    )

    assert state.step_index == 0
    assert environment.last_layout is not None
    assert environment.observe() is observation
    assert stepped.terminated
    environment.close()
    assert calls[0] == ("reset", 61081)
    assert calls[-1] == "close"


def test_environment_rejects_an_invalid_backend() -> None:
    with pytest.raises(TypeError, match="SimulationEnvironment"):
        RedOnBlueEnvironment(backend=object())  # type: ignore[arg-type]


def test_action_tensor_adapter_preserves_batch_shape_and_device() -> None:
    torch = pytest.importorskip("torch")
    like = torch.zeros((2, 3))
    actions = (
        RobotAction(1, 0, 0, 0, -1),
        RobotAction(0, 1, 0, 0, 1),
    )

    tensor = IsaacActionAdapter().to_tensor(actions, like=like)

    assert tensor.shape == (2, 5)
    assert tensor.device == like.device
    assert tensor.tolist() == [[1, 0, 0, 0, -1], [0, 1, 0, 0, 1]]
