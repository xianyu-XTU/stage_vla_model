"""Errors shared by the public V7 boundaries."""


class StageVLAError(Exception):
    """Base class for expected Stage VLA failures."""


class ContractError(StageVLAError, ValueError):
    """Raised when data crossing a module boundary is invalid."""


class ProviderError(StageVLAError, RuntimeError):
    """Raised when a configured provider cannot produce a valid result."""


class RoutingError(StageVLAError, LookupError):
    """Raised when zero or multiple action bundles match a request."""


class UnsupportedTaskError(StageVLAError, ValueError):
    """Raised when a language provider cannot represent a command."""
