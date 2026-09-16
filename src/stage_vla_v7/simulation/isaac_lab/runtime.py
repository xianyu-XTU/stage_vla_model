"""Lifecycle wrapper and explicit adapter for the retained V5 Isaac runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stage_vla_v7.interfaces import (
    SimulationAction,
    SimulationModelDescriptor,
    SimulationObservation,
    SimulationState,
)


class CallbackIsaacLabRuntime:
    """Expose an Isaac application loop through the SimulationEnvironment port."""

    descriptor = SimulationModelDescriptor(
        "isaac-lab-callback-runtime",
        "environment-backend",
        "1",
        "isaac-lab",
        ("explicit-reset", "observe", "step", "close"),
    )

    def __init__(
        self,
        *,
        reset: Callable[[int | None], SimulationState],
        observe: Callable[[], SimulationObservation],
        step: Callable[[SimulationAction], SimulationState],
        close: Callable[[], None],
    ) -> None:
        self._reset = reset
        self._observe = observe
        self._step = step
        self._close = close

    def reset(self, seed: int | None = None) -> SimulationState:
        state = self._reset(seed)
        if not isinstance(state, SimulationState):
            raise TypeError("Isaac reset callback must return SimulationState")
        return state

    def observe(self) -> SimulationObservation:
        observation = self._observe()
        if not isinstance(observation, SimulationObservation):
            raise TypeError("Isaac observe callback must return SimulationObservation")
        return observation

    def step(self, action: SimulationAction) -> SimulationState:
        if not isinstance(action, SimulationAction):
            raise TypeError("Isaac step requires SimulationAction")
        state = self._step(action)
        if not isinstance(state, SimulationState):
            raise TypeError("Isaac step callback must return SimulationState")
        return state

    def close(self) -> None:
        self._close()


def make_legacy_v5_known_size_grasp_env(**kwargs: Any) -> Any:
    """Create the retained V5 physical runtime through the V7 Simulation boundary."""
    try:
        from tools.stageppo_known_size_grasp_env import make_known_size_grasp_env
    except ImportError as exc:
        raise RuntimeError(
            "the retained stage_vla_v5 runtime must be on sys.path before creating "
            "the Isaac Lab environment"
        ) from exc
    return make_known_size_grasp_env(**kwargs)
