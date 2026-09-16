"""Vision package public API."""

from stage_vla_v7.interfaces import VisionProvider, VisionRequest, VisionResult

from .geometry import CameraCalibration
from .providers import (
    CallableVisionProvider,
    CompactColorDepthDetector,
    CompactColorDepthProvider,
    LegacyDetectorAdapter,
    StaticVisionProvider,
)
from .registry import VisionProviderRegistry
from .service import VisionService

__all__ = [
    "CallableVisionProvider",
    "CameraCalibration",
    "CompactColorDepthDetector",
    "CompactColorDepthProvider",
    "LegacyDetectorAdapter",
    "StaticVisionProvider",
    "VisionProvider",
    "VisionProviderRegistry",
    "VisionRequest",
    "VisionResult",
    "VisionService",
]
