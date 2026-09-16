"""Compact vision output contracts; model implementations are intentionally separate."""

from .state import Detection, SceneState, VisionStateAdapter
from .rgbd_detector import CameraCalibration, CompactColorDepthDetector, infer_held_label
from .compact_model import CompactRGBDDetectorNet, LearnedRGBDDetector, decode_positions, pack_rgbd
from .yolo_model import YoloCubeRGBDDetector
from .recognition import (
    CompactRGBDRecognitionBackend,
    LearnedRGBDRecognitionBackend,
    ObjectRecognitionModule,
    RecognitionBackend,
    RecognitionRequest,
    RecognitionResult,
    StaticRecognitionBackend,
    VisionFrame,
)

__all__ = [
    "Detection", "SceneState", "VisionStateAdapter",
    "CameraCalibration", "CompactColorDepthDetector", "infer_held_label",
    "CompactRGBDDetectorNet", "LearnedRGBDDetector", "decode_positions", "pack_rgbd",
    "YoloCubeRGBDDetector",
    "CompactRGBDRecognitionBackend", "LearnedRGBDRecognitionBackend",
    "ObjectRecognitionModule", "RecognitionBackend",
    "RecognitionRequest", "RecognitionResult", "StaticRecognitionBackend", "VisionFrame",
]
