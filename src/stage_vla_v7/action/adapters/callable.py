"""Dependency-free action policies for testing and custom controllers."""

from __future__ import annotations

from collections.abc import Callable

from stage_vla_v7.contracts import ModelDescriptor, RobotAction

from ..interfaces import ActionRequest, ActionResult


class CallableActionPolicy:
    """Wrap a function that maps one validated request to five values."""

    def __init__(
        self,
        function: Callable[[ActionRequest], RobotAction],
        *,
        observation_dim: int,
        name: str,
        version: str = "1",
        capabilities: tuple[str, ...] = (),
    ) -> None:
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        self.function = function
        self.observation_dim = observation_dim
        self.descriptor = ModelDescriptor(name, version, "action", capabilities)

    def predict(self, request: ActionRequest) -> ActionResult:
        action = self.function(request)
        if not isinstance(action, RobotAction):
            raise TypeError("action callable must return RobotAction")
        return ActionResult(action, self.descriptor)


class ConstantActionPolicy(CallableActionPolicy):
    """Simple deterministic policy useful for contract tests."""

    def __init__(self, action: RobotAction, *, observation_dim: int, name: str) -> None:
        super().__init__(lambda _request: action, observation_dim=observation_dim, name=name)
