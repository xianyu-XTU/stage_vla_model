"""Persist whole-task evaluation results independently of video encoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_json_result(path: Path, result: Mapping[str, object]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
