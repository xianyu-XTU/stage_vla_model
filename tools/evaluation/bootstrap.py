"""Make the repository's V7 source package available to evaluation tools."""

from __future__ import annotations

from pathlib import Path
import sys


V7_ROOT = Path(__file__).resolve().parents[2]
V7_SOURCE_ROOT = V7_ROOT / "src"


def ensure_v7_source_available() -> Path:
    """Expose only the local V7 package when the project is not installed."""
    source = str(V7_SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    return V7_SOURCE_ROOT


ensure_v7_source_available()


__all__ = ["V7_ROOT", "V7_SOURCE_ROOT", "ensure_v7_source_available"]
