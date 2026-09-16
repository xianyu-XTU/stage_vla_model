"""Validated loading of external TorchScript runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    sha256: str
    model: Any


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_torchscript_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_sha256: str | None = None,
) -> LoadedCheckpoint:
    """Fail closed on a missing artifact, hash mismatch, or missing Torch runtime."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256.upper():
        raise ValueError(f"checkpoint SHA256 mismatch for {resolved}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TorchScript checkpoints require the 'torch' extra") from exc
    return LoadedCheckpoint(resolved, digest, torch.jit.load(str(resolved), map_location=device).eval())
