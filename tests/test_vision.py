from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.vision import LegacyDetectorAdapter, VisionRequest, VisionService


@dataclass
class LegacyDetection:
    label: str
    position_xyz_m: tuple[float, float, float]
    yaw_rad: float
    confidence: float


class LegacyDetector:
    def detect_one(self, rgb, depth, **kwargs):
        del rgb, depth, kwargs
        return (
            LegacyDetection("red_cube", (0.4, 0.1, 0.02), 0.0, 0.9),
            LegacyDetection("blue_cube", (0.5, 0.0, 0.02), 0.0, 0.8),
        )


def test_legacy_detector_is_normalized_to_v7_contract() -> None:
    provider = LegacyDetectorAdapter(LegacyDetector(), name="legacy", version="5")
    result = VisionService(provider).observe(VisionRequest(rgb=object(), depth_m=object()))
    assert result.scene.detection("red_cube").confidence == 0.9
    assert result.provider.kind == "vision"
