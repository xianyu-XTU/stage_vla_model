"""Compatibility import for the environment factory migrated to V7 Simulation."""

from __future__ import annotations

from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_V7_SOURCE = _REPOSITORY_ROOT / "src"
if _V7_SOURCE.is_dir() and str(_V7_SOURCE) not in sys.path:
    sys.path.insert(0, str(_V7_SOURCE))

from stage_vla_v7.simulation.isaac_lab.env_factory import (  # noqa: E402,F401
    install_known_size_gripper_action,
    make_known_size_grasp_env,
)

__all__ = ["install_known_size_gripper_action", "make_known_size_grasp_env"]
