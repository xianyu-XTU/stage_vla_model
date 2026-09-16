"""Convert an adapter-provided RGB payload into a validated uint8 frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameCapture:
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(int(self.width), int(self.height)) < 1:
            raise ValueError("capture dimensions must be positive")

    def capture(self, payload: object) -> np.ndarray:
        value = payload
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            value = numpy_method()
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("RGB frame must have shape HxWxC with at least 3 channels")
        frame = frame[..., :3]
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                "RGB frame resolution does not match the configured observer camera"
            )
        if np.issubdtype(frame.dtype, np.floating):
            maximum = float(np.nanmax(frame)) if frame.size else 0.0
            if maximum <= 1.0 + 1e-6:
                frame = frame * 255.0
            frame = np.rint(frame)
        if not np.isfinite(frame).all():
            raise ValueError("RGB frame contains non-finite values")
        return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
