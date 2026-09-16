"""Modular action-output boundary shared by skill executors and evaluations.

The module owns the last step between an action source and an environment:
source routing, contract projection, batch masking, and output diagnostics.
Task planning, observations, simulator stepping, and success detection remain
outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import torch

from .rl.skill_action_safety import project_skill_action
from .rl.v5_skill_contracts import ACTION_ORDER, V5Skill, reference_action
from .objects import ObjectModule, ObjectModuleOutput


class ActionSource(Protocol):
    """Minimal interface implemented by learned action-model runtimes."""

    def action(self, skill: V5Skill | str, observation: torch.Tensor) -> torch.Tensor:
        """Return one normalized action per observation."""


class ObjectActionSource(Protocol):
    """Action source conditioned on label-free physical object metadata."""

    def action_for_module(
        self,
        module: ObjectModule,
        skill: V5Skill | str,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Return actions for an object/support pair in a trained domain."""


class MappingActionSource:
    """Adapt a mapping of callables, including TorchScript policies, to ActionSource."""

    def __init__(self, policies: Mapping[V5Skill | str, Callable[[torch.Tensor], object]]) -> None:
        self._policies = {V5Skill(skill): policy for skill, policy in policies.items()}

    def action(self, skill: V5Skill | str, observation: torch.Tensor) -> torch.Tensor:
        skill = V5Skill(skill)
        if skill not in self._policies:
            raise KeyError(f"no action policy configured for {skill.value}")
        obs = torch.as_tensor(observation)
        if obs.ndim != 2 or not torch.isfinite(obs).all():
            raise ValueError("observation must be a finite rank-2 tensor")
        with torch.inference_mode():
            result = self._policies[skill](obs)
        action = torch.as_tensor(result, device=obs.device, dtype=torch.float32)
        expected = (obs.shape[0], len(ACTION_ORDER))
        if action.shape != expected or not torch.isfinite(action).all():
            raise ValueError(f"{skill.value} policy action must be finite with shape {expected}")
        return action


@dataclass(frozen=True)
class ActionOutput:
    """Auditable result of one action-output pass.

    ``candidate`` is the selected source's output before safety projection and
    finished-environment masking. ``command`` is the only tensor intended for
    the environment. ``reference`` is populated only when requested.
    """

    skill: V5Skill
    command: torch.Tensor
    candidate: torch.Tensor
    active: torch.Tensor
    source: str
    reference: torch.Tensor | None = None
    action_order: tuple[str, ...] = ACTION_ORDER


@dataclass(frozen=True)
class ObjectActionOutput:
    """Final result produced jointly by the object and action modules."""

    objects: ObjectModuleOutput
    action: ActionOutput

    @property
    def command(self) -> torch.Tensor:
        return self.action.command

    @property
    def action_context(self) -> dict[str, object]:
        return self.objects.action_context()


class ActionOutputModule:
    """Route and finalize normalized five-dimensional physical actions."""

    def __init__(
        self,
        policy_source: ActionSource | ObjectActionSource | None,
        *,
        reference_skills: tuple[V5Skill | str, ...] = (),
    ) -> None:
        self.policy_source = policy_source
        self.reference_skills = frozenset(V5Skill(skill) for skill in reference_skills)

    def emit(
        self,
        skill: V5Skill | str,
        observation: torch.Tensor,
        *,
        reference_state: Mapping[str, object] | None = None,
        finished: torch.Tensor | None = None,
        translation_limit_m: float = 0.005,
        yaw_limit_rad: float = 0.02,
        include_reference: bool = False,
        object_module: ObjectModule | None = None,
    ) -> ActionOutput:
        """Produce a contract-safe command for a batch of environments."""
        skill = V5Skill(skill)
        obs = torch.as_tensor(observation)
        if obs.ndim != 2 or not torch.isfinite(obs).all():
            raise ValueError("observation must be a finite rank-2 tensor")
        if object_module is not None and not isinstance(object_module, ObjectModule):
            raise TypeError("object_module must be an ObjectModule")
        if finished is None:
            finished_mask = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        else:
            finished_mask = torch.as_tensor(finished, device=obs.device)
            if finished_mask.dtype != torch.bool or finished_mask.shape != (obs.shape[0],):
                raise ValueError(f"finished must be a bool tensor with shape ({obs.shape[0]},)")

        use_reference = skill in self.reference_skills
        needs_reference = use_reference or include_reference
        teacher = None
        if needs_reference:
            if reference_state is None:
                raise ValueError(f"reference_state is required for {skill.value}")
            teacher = reference_action(
                skill,
                reference_state,
                translation_limit_m=translation_limit_m,
                yaw_limit_rad=yaw_limit_rad,
            ).to(device=obs.device, dtype=torch.float32)
            self._validate_action(skill, teacher, obs.shape[0], "reference")

        if use_reference:
            candidate = teacher
            source = "reference"
        else:
            if self.policy_source is None:
                raise RuntimeError(f"no policy action source configured for {skill.value}")
            action_for_module = getattr(self.policy_source, "action_for_module", None)
            if object_module is not None and callable(action_for_module):
                raw_candidate = action_for_module(object_module, skill, obs)
            else:
                action = getattr(self.policy_source, "action", None)
                if not callable(action):
                    raise RuntimeError(
                        "policy action source requires object_module conditioning"
                    )
                raw_candidate = action(skill, obs)
            candidate = torch.as_tensor(
                raw_candidate, device=obs.device, dtype=torch.float32
            )
            self._validate_action(skill, candidate, obs.shape[0], "policy")
            source = "policy"

        candidate = candidate.clone()
        command = project_skill_action(skill.value, candidate)
        command[finished_mask, :4] = 0.0
        command[finished_mask, 4] = self._finished_grip(skill)
        return ActionOutput(
            skill=skill,
            command=command,
            candidate=candidate,
            active=~finished_mask,
            source=source,
            reference=None if teacher is None else teacher.clone(),
        )

    def emit_for(
        self,
        objects: ObjectModule,
        skill: V5Skill | str,
        observation: torch.Tensor,
        **kwargs: object,
    ) -> ObjectActionOutput:
        """Combine a typed object-module result with its finalized action."""
        if not isinstance(objects, ObjectModule):
            raise TypeError("objects must be an ObjectModule")
        if "object_module" in kwargs:
            raise TypeError("emit_for owns the object_module argument")
        action = self.emit(skill, observation, object_module=objects, **kwargs)
        return ObjectActionOutput(objects=objects.output(), action=action)

    @staticmethod
    def _validate_action(skill: V5Skill, action: torch.Tensor, batch: int, label: str) -> None:
        expected = (batch, len(ACTION_ORDER))
        if action.shape != expected or not torch.isfinite(action).all():
            raise ValueError(f"{skill.value} {label} action must be finite with shape {expected}")

    @staticmethod
    def _finished_grip(skill: V5Skill) -> float:
        return 1.0 if skill in {
            V5Skill.REACH, V5Skill.RELEASE_STABILIZE, V5Skill.RETREAT
        } else -1.0
