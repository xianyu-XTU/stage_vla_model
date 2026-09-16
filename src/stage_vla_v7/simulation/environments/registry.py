"""Fail-closed registry for task environments."""

from __future__ import annotations

from stage_vla_v7.interfaces import SimulationEnvironment

from .red_on_blue import RedOnBlueEnvironment


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._environments: dict[str, SimulationEnvironment] = {}

    def register(
        self,
        name: str,
        environment: SimulationEnvironment,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(environment, SimulationEnvironment):
            raise TypeError("environment must implement SimulationEnvironment")
        if environment.descriptor.model_type != "environment":
            raise ValueError("environment descriptor must use model_type 'environment'")
        if name in self._environments and not replace:
            raise ValueError(f"environment {name!r} is already registered")
        self._environments[name] = environment

    def require(self, name: str) -> SimulationEnvironment:
        try:
            return self._environments[name]
        except KeyError as exc:
            raise LookupError(f"unknown environment {name!r}") from exc


def default_environment_registry() -> EnvironmentRegistry:
    registry = EnvironmentRegistry()
    registry.register("red_on_blue", RedOnBlueEnvironment())
    return registry
