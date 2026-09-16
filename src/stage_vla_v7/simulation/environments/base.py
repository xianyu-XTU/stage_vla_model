"""Simulation backend attachment shared by concrete task environments."""

from __future__ import annotations

from stage_vla_v7.interfaces import (
    SimulationAction,
    SimulationEnvironment,
    SimulationObservation,
    SimulationState,
)


class BackendEnvironment:
    """Delegate lifecycle operations to an explicitly attached backend."""

    def __init__(self, backend: SimulationEnvironment | None = None) -> None:
        self._backend: SimulationEnvironment | None = None
        if backend is not None:
            self.attach_backend(backend)

    def attach_backend(self, backend: SimulationEnvironment) -> None:
        if self._backend is not None:
            raise RuntimeError("a simulation backend is already attached")
        if not isinstance(backend, SimulationEnvironment):
            raise TypeError("backend must implement SimulationEnvironment")
        self._backend = backend

    def _require_backend(self) -> SimulationEnvironment:
        if self._backend is None:
            raise RuntimeError("no simulation backend is attached")
        return self._backend

    def reset(self, seed: int | None = None) -> SimulationState:
        return self._require_backend().reset(seed)

    def observe(self) -> SimulationObservation:
        return self._require_backend().observe()

    def step(self, action: SimulationAction) -> SimulationState:
        return self._require_backend().step(action)

    def close(self) -> None:
        self._require_backend().close()
