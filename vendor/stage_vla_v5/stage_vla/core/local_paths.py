"""Small, dependency-free helpers for machine-local paths.

This module intentionally avoids PyYAML because `tools/run_isaaclab.py` is launched
with the user's normal Python before entering the Isaac Lab Python environment.
It only parses simple scalar values under the `paths:` section of config.local.yaml.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def read_local_path(config_file: Path, key: str) -> Path | None:
    """Read a simple scalar path from the ``paths:`` section of a YAML file.

    This is deliberately not a general YAML parser. It supports entries such as::

        paths:
          isaaclab: "E:/work/IsaacLab"

    Args:
        config_file: Path to ``config.local.yaml``.
        key: Key inside the ``paths`` mapping.

    Returns:
        Parsed path, or ``None`` if the file/key is missing.
    """
    if not config_file.exists():
        return None

    text = config_file.read_text(encoding="utf-8")
    in_paths = False
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$")

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if re.match(r"^paths\s*:\s*$", line):
            in_paths = True
            continue

        # A new top-level key ends the paths section.
        if in_paths and not raw_line[:1].isspace() and re.match(r"^[A-Za-z_].*:\s*", line):
            break

        if not in_paths:
            continue

        match = key_pattern.match(line)
        if match:
            value = match.group(1).strip().strip('"\'')
            return Path(value) if value else None

    return None


def resolve_isaaclab_root(project_root: Path) -> Path:
    """Resolve Isaac Lab root from env var first, then local config."""
    env_value = os.environ.get("ISAACLAB_ROOT")
    if env_value:
        root = Path(env_value)
    else:
        root = read_local_path(project_root / "config" / "config.local.yaml", "isaaclab")

    if root is None:
        raise FileNotFoundError(
            "Isaac Lab path is not configured. Set ISAACLAB_ROOT or create "
            "config/config.local.yaml from config.local.yaml.example."
        )

    return root



def resolve_isaac_sim_root(project_root: Path) -> Path:
    """Resolve Isaac Sim standalone root from env var first, then local config.

    Resolution order:
        1. ``ISAAC_SIM_ROOT`` environment variable
        2. ``config/config.local.yaml`` -> ``paths.isaac_sim``
    """
    env_value = os.environ.get("ISAAC_SIM_ROOT")
    if env_value:
        root = Path(env_value)
    else:
        root = read_local_path(project_root / "config" / "config.local.yaml", "isaac_sim")

    if root is None:
        raise FileNotFoundError(
            "Isaac Sim path is not configured. Set ISAAC_SIM_ROOT or add "
            "paths.isaac_sim to config/config.local.yaml."
        )

    return root
