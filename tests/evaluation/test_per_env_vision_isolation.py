from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stage_vla_v7.action import ActionOutputModule, hold_finished_skill_action
from stage_vla_v7.contracts import ObjectDetection
from tools.evaluation.outcomes import (
    build_environment_outcomes,
    build_global_error_outcomes,
)
from tools.evaluation.isolation import VisionIsolation
from tools.evaluation.result_writer import write_json_result
from tools.evaluation.vision_policy import VisionFailPolicy, VisionPositionResolver
from tools.evaluation.vision_runtime import VisionBatchObserver


torch = pytest.importorskip("torch")


def _predictions() -> dict[str, np.ndarray]:
    base = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    return {"cube_1": base.copy(), "cube_2": base.copy() + 1.0}


def test_single_env_strict_vision_failure_only_fails_that_env() -> None:
    resolver = VisionPositionResolver(
        VisionFailPolicy.STRICT, ("cube_1", "cube_2"), 4
    )
    predicted = _predictions()
    predicted["cube_1"][1] = np.nan

    first = resolver.resolve_isolated(predicted)
    assert first.alive_mask.tolist() == [True, False, True, True]
    assert first.newly_failed_mask.tolist() == [False, True, False, False]
    assert np.array_equal(first.positions["cube_1"][1], np.zeros(3))

    second = resolver.resolve_isolated(
        _predictions(), active_mask=first.alive_mask
    )
    assert second.alive_mask.tolist() == [True, False, True, True]
    audit = resolver.audit()
    assert audit["invalid_frames"] == 1
    assert audit["oracle_fallback_count"] == 0
    assert audit["per_environment"][1] == {
        "environment_index": 1,
        "valid": False,
        "invalid_frames": 1,
        "first_invalid_observation": 0,
        "first_invalid_step": 0,
        "failed_objects": ["cube_1"],
        "failure_reason": "invalid_required_detection",
    }


class _RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[torch.Tensor] = []

    def action(self, _skill: object, observation: torch.Tensor) -> torch.Tensor:
        self.observations.append(observation.clone())
        return torch.full((len(observation), 5), 0.5)


def test_peer_envs_continue_and_failed_env_uses_safe_hold() -> None:
    source = _RecordingPolicy()
    module = ActionOutputModule(source)
    observation = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    alive = torch.tensor([True, False, True, True])
    output = module.emit(
        "GRASP",
        observation,
        finished=~alive,
        inference_mask=alive,
    )

    assert len(source.observations) == 1
    assert torch.equal(source.observations[0], observation[[0, 2, 3]])
    assert output.active.tolist() == [True, False, True, True]
    previous = torch.zeros((4, 5), dtype=torch.float32)
    previous[1, 4] = -0.25
    submitted = hold_finished_skill_action(
        "GRASP", output.command, ~alive, previous
    )
    assert submitted[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, -0.25])
    assert module.reference_call_count == 0


def test_failed_env_preserves_gripper_during_later_reach_steps() -> None:
    alive = np.asarray([True, False, True, True])

    def decode(action: torch.Tensor) -> torch.Tensor:
        raw = torch.zeros((len(action), 7))
        raw[:, 6] = 1.0
        return raw

    isolation = VisionIsolation(True, 4, alive, decode)
    previous = torch.zeros((4, 5))
    previous[1, 4] = -0.75
    raw = isolation.reach_raw_action(torch.zeros((4, 5)), previous)

    assert raw[:, 6].tolist() == pytest.approx([1.0, -0.75, 1.0, 1.0])


class _CameraAdapter:
    def rgb_u8_batch(self, _camera: object, *, camera_name: str) -> np.ndarray:
        assert camera_name == "vision"
        return np.zeros((4, 2, 2, 3), dtype=np.uint8)

    def depth_m_batch(self, _camera: object, *, camera_name: str) -> np.ndarray:
        assert camera_name == "vision"
        return np.ones((4, 2, 2), dtype=np.float32)


class _VisionService:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def observe(self, request: object) -> object:
        index = int(request.metadata["environment_index"])
        self.calls.append(index)
        detections = [ObjectDetection("red", (0.1, 0.2, 0.3))]
        if index != 1:
            detections.append(ObjectDetection("blue", (0.4, 0.5, 0.6)))
        return SimpleNamespace(scene=SimpleNamespace(detections=tuple(detections)))


def test_failed_env_is_not_retried_by_vision_runtime() -> None:
    service = _VisionService()
    resolver = VisionPositionResolver(
        VisionFailPolicy.STRICT, ("cube_1", "cube_2"), 4
    )
    stats: dict[str, object] = {
        "frames": 0,
        "v7_service_calls": 0,
        "v7_service_calls_by_environment": [0, 0, 0, 0],
    }
    alive = np.ones(4, dtype=bool)
    observer = VisionBatchObserver(
        camera=object(),
        camera_adapter=_CameraAdapter(),
        service=service,
        resolver=resolver,
        origins=torch.zeros((4, 3)),
        scene_assets=("cube_1", "cube_2"),
        labels={"cube_1": "red", "cube_2": "blue"},
        camera_name="vision",
        device="cpu",
        stats=stats,
        alive=alive,
        debug_oracle_loader=lambda _name: pytest.fail("strict mode touched oracle"),
    )

    observer.observe_positions()
    observer.observe_positions()
    assert service.calls == [0, 1, 2, 3, 0, 2, 3]
    assert stats["v7_service_calls_by_environment"] == [2, 1, 2, 2]
    assert alive.tolist() == [True, False, True, True]


def test_batch_result_contains_independent_environment_outcomes(tmp_path) -> None:
    outcomes = build_environment_outcomes(
        num_envs=4,
        relation_results=(),
        overall_alive=torch.tensor([True, False, True, True]),
        vision={
            "strict_mode": True,
            "oracle_fallback_count": 0,
            "v7_service_calls_by_environment": [5, 1, 5, 5],
            "per_environment": [
                {"environment_index": 0, "valid": True, "invalid_frames": 0},
                {
                    "environment_index": 1,
                    "valid": False,
                    "invalid_frames": 1,
                    "first_invalid_step": 2,
                    "failed_objects": ["cube_2"],
                    "failure_reason": "invalid_required_detection",
                },
                {"environment_index": 2, "valid": True, "invalid_frames": 0},
                {"environment_index": 3, "valid": True, "invalid_frames": 0},
            ],
        },
        runtime_purity={
            "verified": True,
            "vendor_path_exposed": False,
            "loaded_v5_module_count": 0,
            "import_blocker_enabled": True,
        },
        pipeline_audit={"all_prepared_skills_exercised": True},
        reference_skill_calls=0,
        recovery_calls=0,
    )
    path = write_json_result(
        tmp_path / "result.json", {"environment_outcomes": outcomes}
    )

    payload = path.read_text(encoding="utf-8")
    assert '"outcome": "VISION_FAILURE"' in payload
    assert [row["outcome"] for row in outcomes] == [
        "PASS", "VISION_FAILURE", "PASS", "PASS"
    ]
    assert outcomes[1]["first_failure_skill"] == "VISION"
    assert not any(row["peer_aborted_due_to_other_env_failure"] for row in outcomes)


def test_vision_failure_overrides_stale_alive_success() -> None:
    outcomes = build_environment_outcomes(
        num_envs=1,
        relation_results=(),
        overall_alive=torch.tensor([True]),
        vision={
            "strict_mode": True,
            "oracle_fallback_count": 0,
            "v7_service_calls_by_environment": [3],
            "per_environment": [{
                "environment_index": 0,
                "valid": False,
                "invalid_frames": 1,
                "first_invalid_step": 2,
                "failed_objects": ["cube_1"],
                "failure_reason": "invalid_required_detection",
            }],
        },
        runtime_purity={
            "verified": True,
            "vendor_path_exposed": False,
            "loaded_v5_module_count": 0,
            "import_blocker_enabled": True,
        },
        pipeline_audit={"all_prepared_skills_exercised": True},
        reference_skill_calls=0,
        recovery_calls=0,
    )

    assert outcomes[0]["outcome"] == "VISION_FAILURE"
    assert outcomes[0]["first_failure_skill"] == "VISION"
    assert outcomes[0]["physical_success"] is False
    assert outcomes[0]["stable_success"] is False


def test_global_runtime_error_still_aborts_batch() -> None:
    outcomes = build_global_error_outcomes(4, {"message": "CUDA OOM"})
    assert [row["outcome"] for row in outcomes] == ["GLOBAL_RUNTIME_ERROR"] * 4
    assert not any(row["peer_aborted_due_to_other_env_failure"] for row in outcomes)


def test_global_tensor_shape_corruption_is_not_isolated() -> None:
    resolver = VisionPositionResolver(VisionFailPolicy.STRICT, ("cube_1",), 4)
    with pytest.raises(ValueError, match="shape"):
        resolver.resolve_isolated({"cube_1": np.zeros((3, 3), dtype=np.float32)})
