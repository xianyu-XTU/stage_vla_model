"""Video recording configuration and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class RecordingConfig:
    path: Path
    width: int
    height: int
    fps: float
    strict: bool = True
    codec: str = "libx264"
    quality: int = 8

    def __post_init__(self) -> None:
        if min(int(self.width), int(self.height)) < 1:
            raise ValueError("recording dimensions must be positive")
        if not math.isfinite(float(self.fps)) or self.fps <= 0.0:
            raise ValueError("recording fps must be finite and positive")
        if not self.codec.strip():
            raise ValueError("recording codec must be non-empty")
        if not 0 <= int(self.quality) <= 10:
            raise ValueError("recording quality must be in [0,10]")


@dataclass(frozen=True)
class RecordingResult:
    path: Path
    frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    completed: bool
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "frames": self.frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_s": self.duration_s,
            "completed": self.completed,
            "error": self.error,
        }
