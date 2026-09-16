from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from stage_vla.vision.rgbd_detector import (  # noqa: E402
    CameraCalibration as V5CameraCalibration,
)
from stage_vla.vision.rgbd_detector import (  # noqa: E402
    CompactColorDepthDetector as V5CompactColorDepthDetector,
)
from stage_vla_v7.vision import (  # noqa: E402
    CameraCalibration,
    CompactColorDepthDetector,
    CompactColorDepthProvider,
    VisionRequest,
    VisionService,
)


LABELS = ("red_cube", "blue_cube", "green_cube", "yellow_cube")


def _calibration_pair() -> tuple[V5CameraCalibration, CameraCalibration]:
    intrinsic = np.array(
        [[120.0, 0.0, 63.5], [0.0, 121.0, 47.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera_to_root = np.array(
        [
            [0.0, -1.0, 0.0, 0.42],
            [1.0, 0.0, 0.0, -0.03],
            [0.0, 0.0, 1.0, 0.11],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (
        V5CameraCalibration(intrinsic.copy(), camera_to_root.copy()),
        CameraCalibration(intrinsic.copy(), camera_to_root.copy()),
    )


def _synthetic_rgbd() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((96, 128, 4), dtype=np.uint8)
    rgb[..., 3] = 255
    rgb[8:25, 10:31, :3] = (220, 40, 40)
    rgb[28:48, 38:62, :3] = (40, 90, 220)
    rgb[52:75, 68:89, :3] = (40, 180, 80)
    rgb[18:42, 94:119, :3] = (220, 190, 40)
    depth = np.linspace(0.45, 0.95, 96 * 128, dtype=np.float32).reshape(96, 128)
    depth[10, 12] = np.nan
    depth[30, 40] = 0.0
    return rgb, depth[..., None]


def test_native_calibration_matches_v5_scalar_and_vectorized_projection() -> None:
    legacy, native = _calibration_pair()
    pixels = np.array([[0.0, 0.0], [63.5, 47.5], [127.0, 95.0]], dtype=np.float64)
    depths = np.array([0.25, 0.70, 1.25], dtype=np.float64)

    legacy_many = legacy.deproject_pixels(pixels, depths)
    native_many = native.deproject_pixels(pixels, depths)

    assert native_many.shape == legacy_many.shape == (3, 3)
    assert native_many.dtype == legacy_many.dtype == np.float32
    assert np.isfinite(native_many).all()
    assert float(np.max(np.abs(native_many - legacy_many))) == 0.0
    for pixel, depth in zip(pixels, depths):
        assert np.array_equal(
            native.deproject_pixel(pixel, float(depth)),
            legacy.deproject_pixel(pixel, float(depth)),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("intrinsic", np.eye(4), "intrinsic must be"),
        ("camera_to_root", np.eye(3), "camera_to_root must be"),
        ("intrinsic", np.full((3, 3), np.nan), "NaN/Inf"),
    ],
)
def test_native_calibration_rejects_the_same_invalid_shapes(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    _legacy, native = _calibration_pair()
    invalid = CameraCalibration(
        value if field == "intrinsic" else native.intrinsic,
        value if field == "camera_to_root" else native.camera_to_root,
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_native_detector_matches_v5_for_all_runtime_cube_labels() -> None:
    legacy_calibration, native_calibration = _calibration_pair()
    rgb, depth = _synthetic_rgbd()
    kwargs = {
        "min_pixels": 12,
        "confidence_floor": 0.20,
        "position_bias_m": (-0.0099, 0.00214, 0.0),
    }
    legacy = V5CompactColorDepthDetector(**kwargs)
    native = CompactColorDepthDetector(**kwargs)

    expected = legacy.detect_scene(
        rgb,
        depth,
        calibration=legacy_calibration,
        labels=LABELS,
    )
    actual = native.detect_scene(
        rgb,
        depth,
        calibration=native_calibration,
        labels=LABELS,
    )

    assert tuple(item.label for item in actual) == tuple(item.label for item in expected) == LABELS
    for native_item, legacy_item in zip(actual, expected):
        assert np.array_equal(native_item.position_xyz_m, legacy_item.position_xyz_m)
        assert native_item.yaw_rad == legacy_item.yaw_rad
        assert native_item.confidence == legacy_item.confidence
        assert np.isfinite((*native_item.position_xyz_m, native_item.yaw_rad)).all()


@pytest.mark.parametrize("label", LABELS)
def test_native_detector_matches_v5_for_missing_and_partial_depth(label: str) -> None:
    legacy_calibration, native_calibration = _calibration_pair()
    rgb, depth = _synthetic_rgbd()
    depth[:] = np.nan
    legacy = V5CompactColorDepthDetector(min_pixels=4)
    native = CompactColorDepthDetector(min_pixels=4)

    expected = legacy.detect_one(
        rgb,
        depth,
        label=label,
        calibration=legacy_calibration,
    )
    actual = native.detect_one(
        rgb,
        depth,
        label=label,
        calibration=native_calibration,
    )

    assert actual is expected is None


def test_native_provider_emits_v7_scene_without_legacy_adapter() -> None:
    _legacy_calibration, calibration = _calibration_pair()
    rgb, depth = _synthetic_rgbd()
    provider = CompactColorDepthProvider(
        calibration=calibration,
        labels=LABELS,
        detector=CompactColorDepthDetector(min_pixels=12),
    )

    result = VisionService(provider).observe(
        VisionRequest(rgb, depth, frame_id="synthetic-rgbd", timestamp_s=1.25)
    )

    assert tuple(item.label for item in result.scene.detections) == LABELS
    assert result.scene.frame_id == "synthetic-rgbd"
    assert result.scene.timestamp_s == 1.25
    assert result.provider.name == "v7-compact-color-depth"
    assert result.diagnostics["used_fallback"] is False


def test_physical_runtime_uses_native_detector_and_calibration() -> None:
    source = (ROOT / "tools" / "evaluation" / "episode_runner.py").read_text(
        encoding="utf-8"
    )

    assert "from stage_vla.vision" not in source
    assert "LegacyDetectorAdapter" not in source
    assert "CompactColorDepthProvider" in source
