from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from stage_vla.data.v4_2_runtime import (  # noqa: E402
    read_depth_m_batch as v5_read_depth_m_batch,
    read_rgb_u8_batch as v5_read_rgb_u8_batch,
)
from stage_vla_v7.simulation.isaac_lab import IsaacCameraAdapter  # noqa: E402


CAMERA_NAME = "parity_camera"


class _ProxyArray:
    def __init__(self, value: object) -> None:
        self.torch = value


def _camera_env(**outputs: object) -> tuple[object, object]:
    camera = SimpleNamespace(data=SimpleNamespace(output=outputs))
    return SimpleNamespace(scene={CAMERA_NAME: camera}), camera


@pytest.mark.parametrize(
    "payload",
    [
        torch.arange(2 * 5 * 7 * 4, dtype=torch.uint8).reshape(2, 5, 7, 4),
        torch.linspace(0.0, 1.0, 2 * 5 * 7 * 3, dtype=torch.float32).reshape(
            2, 5, 7, 3
        ),
        torch.linspace(-10.0, 300.0, 2 * 5 * 7 * 4, dtype=torch.float64).reshape(
            2, 5, 7, 4
        ),
    ],
    ids=["uint8_rgba", "normalized_float_rgb", "float_rgba_clamped"],
)
@pytest.mark.parametrize("proxied", [False, True], ids=["tensor", "proxy"])
def test_native_rgb_reader_matches_v5(payload: object, proxied: bool) -> None:
    source = _ProxyArray(payload) if proxied else payload
    base_env, camera = _camera_env(rgb=source)

    expected = v5_read_rgb_u8_batch(base_env, camera_name=CAMERA_NAME)
    actual = IsaacCameraAdapter().rgb_u8_batch(camera, camera_name=CAMERA_NAME)

    assert np.array_equal(actual, expected)
    assert actual.shape == expected.shape == (2, 5, 7, 3)
    assert actual.dtype == expected.dtype == np.uint8
    assert actual.flags.c_contiguous and expected.flags.c_contiguous
    assert int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))) == 0


@pytest.mark.parametrize(
    "payload",
    [
        torch.linspace(0.1, 2.0, 2 * 5 * 7, dtype=torch.float64).reshape(2, 5, 7),
        torch.linspace(0.1, 2.0, 2 * 5 * 7, dtype=torch.float64).reshape(2, 5, 7, 1),
    ],
    ids=["nhw", "nhw1"],
)
@pytest.mark.parametrize("proxied", [False, True], ids=["tensor", "proxy"])
def test_native_depth_reader_matches_v5(payload: object, proxied: bool) -> None:
    source = _ProxyArray(payload) if proxied else payload
    base_env, camera = _camera_env(distance_to_image_plane=source)

    expected = v5_read_depth_m_batch(base_env, camera_name=CAMERA_NAME)
    actual = IsaacCameraAdapter().depth_m_batch(camera, camera_name=CAMERA_NAME)

    assert np.array_equal(actual, expected)
    assert actual.shape == expected.shape == (2, 5, 7)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.flags.c_contiguous and expected.flags.c_contiguous
    assert np.isfinite(actual).all()
    assert float(np.max(np.abs(actual - expected))) == 0.0


@pytest.mark.parametrize(
    ("method", "outputs", "message"),
    [
        ("rgb_u8_batch", {}, "has no rgb output"),
        ("rgb_u8_batch", {"rgb": torch.zeros((2, 5, 7))}, "unexpected batched camera RGB"),
        ("depth_m_batch", {}, "has no distance_to_image_plane output"),
        (
            "depth_m_batch",
            {"distance_to_image_plane": torch.zeros((2, 5, 7, 2))},
            "unexpected batched camera depth",
        ),
    ],
)
def test_native_camera_readers_fail_closed(
    method: str,
    outputs: dict[str, object],
    message: str,
) -> None:
    _base_env, camera = _camera_env(**outputs)

    with pytest.raises(RuntimeError, match=message):
        getattr(IsaacCameraAdapter(), method)(camera, camera_name=CAMERA_NAME)


def test_physical_runtime_uses_native_camera_readers() -> None:
    source = (ROOT / "tools" / "evaluation" / "episode_runner.py").read_text(
        encoding="utf-8"
    )

    assert "stage_vla.data.v4_2_runtime" not in source
    assert "camera_adapter.rgb_u8_batch" in source
    assert "camera_adapter.depth_m_batch" in source
