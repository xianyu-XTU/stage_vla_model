"""Explicit, regression-only access to the retained V5 reference source."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENDORED_V5_REFERENCE_ROOT = REPOSITORY_ROOT / "vendor" / "stage_vla_v5"


def resolve_v5_reference_root(root: str | Path | None = None) -> Path:
    """Resolve an explicitly requested V5 reference tree for migration work."""
    configured = root if root is not None else os.environ.get("STAGE_VLA_V5_ROOT")
    resolved = Path(configured or VENDORED_V5_REFERENCE_ROOT).resolve()
    if not (resolved / "stage_vla" / "__init__.py").is_file():
        raise FileNotFoundError(f"V5 reference package is unavailable: {resolved}")
    return resolved


@contextmanager
def with_v5_reference_path(
    root: str | Path | None = None,
) -> Iterator[Path]:
    """Temporarily expose V5 for an explicit parity or migration operation."""
    resolved = resolve_v5_reference_root(root)
    source = str(resolved)
    inserted = source not in sys.path
    if inserted:
        sys.path.insert(0, source)
    try:
        yield resolved
    finally:
        if inserted:
            try:
                sys.path.remove(source)
            except ValueError:  # pragma: no cover - caller mutated sys.path
                pass


__all__ = [
    "VENDORED_V5_REFERENCE_ROOT",
    "resolve_v5_reference_root",
    "with_v5_reference_path",
]
