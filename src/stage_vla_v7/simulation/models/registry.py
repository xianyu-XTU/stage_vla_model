"""Fail-closed registry for robot, object, and sensor model descriptions."""

from __future__ import annotations

from typing import Generic, TypeVar

from stage_vla_v7.interfaces import SimulationModelDescriptor

from .objects import CubeModel
from .robots import FrankaModel
from .sensors import DepthCameraModel, RGBCameraModel


ModelT = TypeVar("ModelT")


class _ModelStore(Generic[ModelT]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._values: dict[str, ModelT] = {}

    def register(self, name: str, model: ModelT, *, replace: bool = False) -> None:
        if not name.strip():
            raise ValueError(f"{self.kind} name must be non-empty")
        descriptor = getattr(model, "descriptor", None)
        if not isinstance(descriptor, SimulationModelDescriptor):
            raise TypeError(f"{self.kind} model must expose SimulationModelDescriptor")
        if descriptor.model_type != self.kind:
            raise ValueError(
                f"{self.kind} registry cannot accept model type {descriptor.model_type!r}"
            )
        if name in self._values and not replace:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        self._values[name] = model

    def require(self, name: str) -> ModelT:
        try:
            return self._values[name]
        except KeyError as exc:
            raise LookupError(f"unknown {self.kind} model {name!r}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))


class SimulationModelRegistry:
    def __init__(self) -> None:
        self.robots = _ModelStore[object]("robot")
        self.objects = _ModelStore[object]("object")
        self.sensors = _ModelStore[object]("sensor")

    def register_robot(self, name: str, model: object, *, replace: bool = False) -> None:
        self.robots.register(name, model, replace=replace)

    def register_object(self, name: str, model: object, *, replace: bool = False) -> None:
        self.objects.register(name, model, replace=replace)

    def register_sensor(self, name: str, model: object, *, replace: bool = False) -> None:
        self.sensors.register(name, model, replace=replace)

    def require_robot(self, name: str) -> object:
        return self.robots.require(name)

    def require_object(self, name: str) -> object:
        return self.objects.require(name)

    def require_sensor(self, name: str) -> object:
        return self.sensors.require(name)


def default_model_registry() -> SimulationModelRegistry:
    registry = SimulationModelRegistry()
    registry.register_robot("franka", FrankaModel())
    registry.register_object("cube", CubeModel())
    registry.register_sensor("rgb", RGBCameraModel())
    registry.register_sensor("depth", DepthCameraModel())
    return registry
