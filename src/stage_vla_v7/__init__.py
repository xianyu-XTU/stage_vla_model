"""Public composition surface for Stage VLA V7."""

from . import action, contracts, language, orchestration, plugins, vision
from .orchestration import PreparedTask, StageVLAPipeline

__version__ = "0.1.0"

__all__ = [
    "PreparedTask",
    "StageVLAPipeline",
    "action",
    "contracts",
    "language",
    "orchestration",
    "plugins",
    "vision",
]
