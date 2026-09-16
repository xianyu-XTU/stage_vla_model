"""Public composition surface for Stage VLA V7."""

from . import action, contracts, interfaces, language, orchestration, plugins, vision
from .orchestration import PreparedTask, StageVLAPipeline

__version__ = "7.0.0"

__all__ = [
    "PreparedTask",
    "StageVLAPipeline",
    "action",
    "contracts",
    "interfaces",
    "language",
    "orchestration",
    "plugins",
    "vision",
]
