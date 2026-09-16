"""Repository paths required by the physical evaluation adapters."""

from __future__ import annotations

import os
from pathlib import Path
import sys


V7_ROOT = Path(__file__).resolve().parents[2]
VENDORED_V5_ROOT = V7_ROOT / "vendor" / "stage_vla_v5"
V5_ROOT = Path(
    os.environ.get(
        "STAGE_VLA_V5_ROOT",
        VENDORED_V5_ROOT if VENDORED_V5_ROOT.is_dir() else V7_ROOT.parent / "stage_vla_v5",
    )
).resolve()

for source_root in (V5_ROOT, V7_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


__all__ = ["V5_ROOT", "V7_ROOT", "VENDORED_V5_ROOT"]
