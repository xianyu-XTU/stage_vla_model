"""Application-layer example for wrapping the V5 YOLO RGB-D detector."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from stage_vla_v7.vision import LegacyDetectorAdapter, VisionService


V5_ROOT = Path("E:/stage_vla_v5")
CHECKPOINT = V5_ROOT / "outputs/yolo26n_cube_seg_pilot_v2/weights/best.pt"
CALIBRATION = V5_ROOT / "outputs/vision_rgbd_mapping_calibration_train_20260910.json"


def build_vision_service(device: str = "cuda:0") -> VisionService:
    """Create V5 dependencies only at the application composition root."""
    sys.path.insert(0, str(V5_ROOT))
    from stage_vla.vision import CameraCalibration, YoloCubeRGBDDetector

    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    calibration = CameraCalibration(
        intrinsic=np.asarray(payload["intrinsic"], dtype=np.float64),
        camera_to_root=np.asarray(payload["camera_to_root"], dtype=np.float64),
    )
    detector = YoloCubeRGBDDetector(CHECKPOINT, device=device)
    provider = LegacyDetectorAdapter(
        detector,
        name="v5-yolo-cube-rgbd",
        version="pilot-v2",
        method_kwargs={"calibration": calibration},
    )
    return VisionService(provider)


if __name__ == "__main__":
    print(build_vision_service().provider.descriptor)
