"""Dependency-injected action policies used for tests and custom networks."""

from __future__ import annotations

from collections.abc import Callable

from stage_vla_v7.interfaces import (
    ActionRequest,
    ActionResult,
    ModelDescriptor,
    RobotAction,
)


class CallableActionPolicy:
    def __init__(
        self,
        function: Callable[[ActionRequest], RobotAction],
        *,
        observation_dim: int,
        name: str,
        version: str = "1",
    ) -> None:
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        self.function = function
        self.observation_dim = observation_dim
        self.descriptor = ModelDescriptor(name, version, "action", ("callable",))

    def predict(self, request: ActionRequest) -> ActionResult:
        action = self.function(request)
        if not isinstance(action, RobotAction):
            raise TypeError("action callable must return RobotAction")
        return ActionResult(action, self.descriptor)


class ConstantActionPolicy(CallableActionPolicy):
    def __init__(self, action: RobotAction, *, observation_dim: int, name: str) -> None:
        super().__init__(
            lambda _request: action,
            observation_dim=observation_dim,
            name=name,
        )
