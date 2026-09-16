"""Observer-only video capture and encoding."""

from .config import RecordingConfig, RecordingResult
from .frame_capture import FrameCapture
from .overlays import draw_overlay
from .video_recorder import VideoRecorder

__all__ = [
    "FrameCapture",
    "RecordingConfig",
    "RecordingResult",
    "VideoRecorder",
    "draw_overlay",
]
