"""Simulation task environments."""

from .base import BackendEnvironment
from .align_reward import AlignRewardConfig, align_reward_terms
from .known_size_observation import (
    LEGACY_OBSERVATION_DIM,
    KnownSizeObservationInput,
    build_known_size_observation,
)
from .red_on_blue import RedOnBlueEnvironment
from .reach_observation import REACH_OBS_DIM, reach_observation
from .registry import EnvironmentRegistry, default_environment_registry
from .transport_handoff import (
    TransportHandoffConfig,
    transport_handoff_ready,
    transport_settle_reward_terms,
)

__all__ = [
    "BackendEnvironment",
    "AlignRewardConfig",
    "EnvironmentRegistry",
    "KnownSizeObservationInput",
    "LEGACY_OBSERVATION_DIM",
    "RedOnBlueEnvironment",
    "REACH_OBS_DIM",
    "TransportHandoffConfig",
    "align_reward_terms",
    "default_environment_registry",
    "reach_observation",
    "build_known_size_observation",
    "transport_handoff_ready",
    "transport_settle_reward_terms",
]
