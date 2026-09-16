from __future__ import annotations

import numpy as np
import pytest

from tools.evaluation.audit import verify_v7_chain
from tools.evaluation.vision_policy import (
    VisionDetectionError,
    VisionFailPolicy,
    VisionPositionResolver,
)


ASSETS = ("cube_1", "cube_2")


def _missing_prediction() -> dict[str, np.ndarray]:
    return {
        "cube_1": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "cube_2": np.full((1, 3), np.nan, dtype=np.float32),
    }


def test_strict_vision_never_reads_oracle() -> None:
    resolver = VisionPositionResolver(VisionFailPolicy.STRICT, ASSETS, 1)
    oracle_calls = 0

    def oracle_loader(_asset_name: str) -> np.ndarray:
        nonlocal oracle_calls
        oracle_calls += 1
        return np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(VisionDetectionError):
        resolver.resolve(_missing_prediction(), oracle_loader=oracle_loader)

    assert oracle_calls == 0


def test_strict_vision_invalid_detection_fails() -> None:
    resolver = VisionPositionResolver(VisionFailPolicy.STRICT, ASSETS, 1)

    with pytest.raises(VisionDetectionError) as failure:
        resolver.resolve(_missing_prediction())

    assert failure.value.failed_objects == ("cube_2",)
    assert failure.value.failed_environments == (0,)
    assert resolver.audit()["invalid_frames"] == 1


def test_debug_oracle_can_fallback() -> None:
    resolver = VisionPositionResolver(VisionFailPolicy.DEBUG_ORACLE, ASSETS, 1)

    positions, valid = resolver.resolve(
        _missing_prediction(),
        oracle_loader=lambda _asset: np.array([[0.4, 0.5, 0.6]], dtype=np.float32),
    )

    assert valid.tolist() == [True]
    assert np.allclose(positions["cube_2"], [[0.4, 0.5, 0.6]])


def test_oracle_fallback_is_audited() -> None:
    resolver = VisionPositionResolver(VisionFailPolicy.DEBUG_ORACLE, ASSETS, 1)
    resolver.resolve(
        _missing_prediction(),
        oracle_loader=lambda _asset: np.array([[0.4, 0.5, 0.6]], dtype=np.float32),
    )

    audit = resolver.audit()
    assert audit["strict_mode"] is False
    assert audit["oracle_fallback_used"] is True
    assert audit["oracle_fallback_count"] == 1
    assert audit["oracle_fallback_objects"] == ["cube_2"]
    assert audit["oracle_fallback_steps"] == [0]
    assert audit["failed_objects"] == ["cube_2"]
    assert audit["invalid_frames"] == 1


def test_require_v7_chain_rejects_oracle_fallback() -> None:
    verified = verify_v7_chain(
        use_vision=True,
        learned_reach=True,
        reference_skills=(),
        reach_reference_recovery_used=False,
        vision={
            "v7_service_calls": 8,
            "invalid_frames": 1,
            "oracle_fallback_count": 1,
        },
        pipeline_audit={"all_prepared_skills_exercised": True},
    )

    assert verified is False
