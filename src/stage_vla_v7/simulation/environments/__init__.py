"""Simulation task environments."""

from .base import BackendEnvironment
from .red_on_blue import RedOnBlueEnvironment
from .registry import EnvironmentRegistry, default_environment_registry

__all__ = [
    "BackendEnvironment",
    "EnvironmentRegistry",
    "RedOnBlueEnvironment",
    "default_environment_registry",
]
