"""Validated action routing, inference, and safety projection."""

from __future__ import annotations

from stage_vla_v7.contracts import ProviderError

from .domain import ActionRouter
from .interfaces import ActionRequest, ActionResult
from .safety import SafetyProjector


class ActionService:
    """The sole V7 boundary allowed to produce a continuous robot action."""

    def __init__(self, router: ActionRouter, *, safety: SafetyProjector | None = None) -> None:
        self.router = router
        self.safety = safety or SafetyProjector()

    def act(self, request: ActionRequest) -> ActionResult:
        bundle = self.router.route(request.object_profile, request.support_profile)
        policy = bundle.policy(request.skill)
        if len(request.observation) != policy.observation_dim:
            raise ValueError(
                f"{request.skill.value} observation must have {policy.observation_dim} values; "
                f"received {len(request.observation)}"
            )
        result = policy.predict(request)
        if not isinstance(result, ActionResult):
            raise ProviderError("action policy returned a non-ActionResult value")
        if result.provider != policy.descriptor:
            raise ProviderError("action result descriptor does not match selected policy")
        projected = self.safety.project(request.skill, result.action, finished=request.finished)
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "bundle": bundle.descriptor.name,
                "safety_projected": projected != result.action,
                "finished": request.finished,
            }
        )
        return ActionResult(projected, result.provider, diagnostics)
