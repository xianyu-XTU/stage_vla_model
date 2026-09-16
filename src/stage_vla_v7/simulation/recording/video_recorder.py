"""Streaming observer-video recorder independent of Isaac Lab tensors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import warnings

import numpy as np

from .config import RecordingConfig, RecordingResult
from .overlays import draw_overlay


WriterFactory = Callable[..., object]


class VideoRecorder:
    def __init__(
        self,
        config: RecordingConfig,
        *,
        writer_factory: WriterFactory | None = None,
    ) -> None:
        self.config = config
        self._writer_factory = writer_factory
        self._writer: object | None = None
        self._started = False
        self._stopped = False
        self._frames = 0
        self._error: str | None = None

    @property
    def active(self) -> bool:
        return self._started and not self._stopped and self._writer is not None

    def _handle_error(self, exc: Exception) -> None:
        self._error = f"{type(exc).__name__}: {exc}"
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                getattr(writer, "close")()
            except Exception:
                pass
        if self.config.strict:
            raise exc
        warnings.warn(f"observer recording disabled: {self._error}", RuntimeWarning)

    def fail(self, exc: Exception) -> None:
        """Apply the configured failure policy to an upstream capture error."""
        if not self._started or self._stopped:
            raise RuntimeError("VideoRecorder.fail() requires an active recording")
        self._handle_error(exc)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("VideoRecorder.start() may only be called once")
        self._started = True
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            factory = self._writer_factory
            if factory is None:
                import imageio.v2 as imageio

                factory = imageio.get_writer
            self._writer = factory(
                self.config.path,
                fps=self.config.fps,
                codec=self.config.codec,
                quality=self.config.quality,
                macro_block_size=None,
            )
        except Exception as exc:
            self._handle_error(exc)

    def capture(
        self,
        *,
        frame: np.ndarray,
        step: int,
        skill: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not self._started or self._stopped:
            raise RuntimeError("VideoRecorder.capture() requires an active recording")
        if self._writer is None:
            return
        try:
            if np.asarray(frame).shape != (self.config.height, self.config.width, 3):
                raise ValueError("recording frame resolution does not match RecordingConfig")
            details = dict(metadata or {})
            rendered = draw_overlay(
                np.asarray(frame, dtype=np.uint8),
                step=step,
                skill=skill,
                seed=details.pop("seed", None),
                env_id=details.pop("env_id", None),
                metadata=details,
            )
            append_data = getattr(self._writer, "append_data")
            append_data(rendered)
            self._frames += 1
        except Exception as exc:
            self._handle_error(exc)

    def stop(self) -> RecordingResult:
        if not self._started:
            raise RuntimeError("VideoRecorder.stop() called before start()")
        if self._stopped:
            raise RuntimeError("VideoRecorder.stop() may only be called once")
        self._stopped = True
        if self._writer is not None:
            writer = self._writer
            self._writer = None
            try:
                close = getattr(writer, "close")
                close()
            except Exception as exc:
                self._handle_error(exc)
        completed = self._error is None and self._frames > 0 and self.config.path.is_file()
        if self.config.strict and self._frames == 0:
            raise RuntimeError("strict observer recording captured no frames")
        if self.config.strict and not self.config.path.is_file():
            raise RuntimeError(f"recording output was not created: {self.config.path}")
        return RecordingResult(
            path=self.config.path,
            frames=self._frames,
            fps=float(self.config.fps),
            width=int(self.config.width),
            height=int(self.config.height),
            duration_s=self._frames / float(self.config.fps),
            completed=completed,
            error=self._error,
        )
