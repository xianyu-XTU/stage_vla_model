"""Dependency-free errors shared by all Stage VLA boundaries."""


class StageVLAError(Exception):
    """Base class for expected Stage VLA failures."""


class ContractError(StageVLAError, ValueError):
    """Raised when data crossing a module boundary is invalid."""


class ProviderError(StageVLAError, RuntimeError):
    """Raised when a configured provider cannot produce a valid result."""


class RoutingError(StageVLAError, LookupError):
    """Raised when zero or multiple registered implementations match."""


class UnsupportedTaskError(StageVLAError, ValueError):
    """Raised when a language provider cannot represent a command."""
