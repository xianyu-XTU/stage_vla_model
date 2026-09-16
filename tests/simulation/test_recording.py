from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stage_vla_v7.simulation.recording import (
    FrameCapture,
    RecordingConfig,
    VideoRecorder,
    draw_overlay,
)


class _FakeWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame).copy())

    def close(self) -> None:
        self.closed = True
        self.path.write_bytes(b"fake-mp4")


class _FailingWriter(_FakeWriter):
    def append_data(self, frame: np.ndarray) -> None:
        raise OSError("encoder rejected frame")


def _recorder(tmp_path: Path, *, fps: float = 20.0):
    state: dict[str, object] = {}

    def factory(path: Path, **_kwargs: object) -> _FakeWriter:
        writer = _FakeWriter(path)
        state["writer"] = writer
        return writer

    output = tmp_path / "episode.mp4"
    recorder = VideoRecorder(
        RecordingConfig(output, width=64, height=48, fps=fps),
        writer_factory=factory,
    )
    return recorder, output, state


def test_video_recorder_start_stop(tmp_path: Path) -> None:
    recorder, _output, _state = _recorder(tmp_path)
    recorder.start()
    recorder.capture(
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        step=1,
        skill="REACH",
    )
    result = recorder.stop()
    assert result.completed


def test_video_output_exists(tmp_path: Path) -> None:
    recorder, output, _state = _recorder(tmp_path)
    recorder.start()
    recorder.capture(
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        step=1,
        skill="GRASP",
    )
    recorder.stop()
    assert output.is_file()


def test_frame_count(tmp_path: Path) -> None:
    recorder, _output, _state = _recorder(tmp_path)
    recorder.start()
    for step in range(3):
        recorder.capture(
            frame=np.zeros((48, 64, 3), dtype=np.uint8),
            step=step,
            skill="LIFT",
        )
    assert recorder.stop().frames == 3


def test_video_resolution(tmp_path: Path) -> None:
    recorder, _output, state = _recorder(tmp_path)
    recorder.start()
    recorder.capture(
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        step=1,
        skill="TRANSPORT",
    )
    result = recorder.stop()
    writer = state["writer"]
    assert isinstance(writer, _FakeWriter)
    assert writer.frames[0].shape == (48, 64, 3)
    assert (result.width, result.height) == (64, 48)


def test_video_fps(tmp_path: Path) -> None:
    recorder, _output, _state = _recorder(tmp_path, fps=25.0)
    recorder.start()
    recorder.capture(
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        step=1,
        skill="ALIGN",
    )
    result = recorder.stop()
    assert result.fps == 25.0
    assert result.duration_s == 1 / 25.0


def test_overlay() -> None:
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    rendered = draw_overlay(
        source,
        step=7,
        skill="DESCEND",
        seed=2012,
        env_id=0,
    )
    assert rendered.shape == source.shape
    assert np.any(rendered != source)


def test_frame_capture_normalizes_float_rgba() -> None:
    payload = np.ones((48, 64, 4), dtype=np.float32) * 0.5
    frame = FrameCapture(64, 48).capture(payload)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 0]) == 128


def test_recording_best_effort_reports_and_closes_encoder_error(tmp_path: Path) -> None:
    output = tmp_path / "best-effort.mp4"
    writer = _FailingWriter(output)
    recorder = VideoRecorder(
        RecordingConfig(output, width=64, height=48, fps=20.0, strict=False),
        writer_factory=lambda *_args, **_kwargs: writer,
    )
    recorder.start()
    with pytest.warns(RuntimeWarning, match="observer recording disabled"):
        recorder.capture(
            frame=np.zeros((48, 64, 3), dtype=np.uint8),
            step=1,
            skill="REACH",
        )
    result = recorder.stop()
    assert writer.closed
    assert not result.completed
    assert result.frames == 0
    assert "encoder rejected frame" in (result.error or "")


def test_recording_strict_raises_and_closes_encoder_error(tmp_path: Path) -> None:
    output = tmp_path / "strict.mp4"
    writer = _FailingWriter(output)
    recorder = VideoRecorder(
        RecordingConfig(output, width=64, height=48, fps=20.0, strict=True),
        writer_factory=lambda *_args, **_kwargs: writer,
    )
    recorder.start()
    with pytest.raises(OSError, match="encoder rejected frame"):
        recorder.capture(
            frame=np.zeros((48, 64, 3), dtype=np.uint8),
            step=1,
            skill="REACH",
        )
    assert writer.closed


def test_recording_best_effort_absorbs_upstream_capture_error(tmp_path: Path) -> None:
    recorder = VideoRecorder(
        RecordingConfig(
            tmp_path / "capture-error.mp4",
            width=64,
            height=48,
            fps=20.0,
            strict=False,
        ),
        writer_factory=lambda path, **_kwargs: _FakeWriter(path),
    )
    recorder.start()
    with pytest.warns(RuntimeWarning, match="observer recording disabled"):
        recorder.fail(ValueError("invalid observer payload"))
    result = recorder.stop()
    assert not result.completed
    assert "invalid observer payload" in (result.error or "")


def test_recording_best_effort_absorbs_output_directory_error(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"keep-existing-file")
    recorder = VideoRecorder(
        RecordingConfig(
            blocker / "episode.mp4",
            width=64,
            height=48,
            fps=20.0,
            strict=False,
        ),
    )
    with pytest.warns(RuntimeWarning, match="observer recording disabled"):
        recorder.start()
    result = recorder.stop()
    assert not recorder.active
    assert not result.completed
    assert result.error
    assert blocker.read_bytes() == b"keep-existing-file"
