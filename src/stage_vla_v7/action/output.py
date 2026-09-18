"""Final batched 5D action selection, projection, masking, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from stage_vla_v7.interfaces import Skill

from .reference import reference_action
from .safety import project_skill_action


ACTION_ORDER = ("dx", "dy", "dz", "dyaw", "grip")


class BatchedActionSource(Protocol):
    def action(self, skill: Skill | str, observation: object) -> object: ...


@dataclass(frozen=True)
class ActionOutput:
    skill: Skill
    command: object
    candidate: object
    active: object
    source: str
    reference: object | None = None
    action_order: tuple[str, ...] = ACTION_ORDER


class ActionOutputModule:
    """Finalize normalized policy or diagnostic reference actions for Isaac."""

    def __init__(
        self,
        policy_source: BatchedActionSource | None,
        *,
        reference_skills: tuple[Skill | str, ...] = (),
    ) -> None:
        self.policy_source = policy_source
        self.reference_skills = frozenset(
            Skill(getattr(skill, "value", skill)) for skill in reference_skills
        )
        self.reference_call_count = 0

    def emit(
        self,
        skill: Skill | str,
        observation: object,
        *,
        reference_state: Mapping[str, object] | None = None,
        finished: object | None = None,
        inference_mask: object | None = None,
        translation_limit_m: float = 0.005,
        yaw_limit_rad: float = 0.02,
        include_reference: bool = False,
    ) -> ActionOutput:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - tensor bridge requires torch
            raise RuntimeError("batched action output requires torch") from exc

        canonical = Skill(getattr(skill, "value", skill))
        obs = torch.as_tensor(observation)
        if obs.ndim != 2 or not torch.isfinite(obs).all():
            raise ValueError("observation must be a finite rank-2 tensor")
        if finished is None:
            finished_mask = torch.zeros(
                obs.shape[0], dtype=torch.bool, device=obs.device
            )
        else:
            finished_mask = torch.as_tensor(finished, device=obs.device)
            if finished_mask.dtype != torch.bool or finished_mask.shape != (obs.shape[0],):
                raise ValueError(
                    f"finished must be a bool tensor with shape ({obs.shape[0]},)"
                )
        isolate_inference = inference_mask is not None
        if not isolate_inference:
            inference = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        else:
            inference = torch.as_tensor(inference_mask, device=obs.device)
            if inference.dtype != torch.bool or inference.shape != (obs.shape[0],):
                raise ValueError(
                    f"inference_mask must be a bool tensor with shape ({obs.shape[0]},)"
                )
        active = inference & ~finished_mask
        inference_rows = active if isolate_inference else inference

        use_reference = canonical in self.reference_skills
        teacher = None
        selected_teacher = None
        if (use_reference or include_reference) and bool(inference_rows.any()):
            if reference_state is None:
                raise ValueError(f"reference_state is required for {canonical.value}")
            selected_state = {
                name: self._select_batch_rows(value, inference_rows, obs.shape[0])
                for name, value in reference_state.items()
            }
            selected_teacher = reference_action(
                canonical,
                selected_state,
                translation_limit_m=translation_limit_m,
                yaw_limit_rad=yaw_limit_rad,
            ).to(device=obs.device, dtype=torch.float32)
            self._validate_action(
                canonical, selected_teacher, int(inference_rows.sum()), "reference"
            )
            teacher = torch.zeros(
                (obs.shape[0], len(ACTION_ORDER)),
                dtype=torch.float32,
                device=obs.device,
            )
            teacher[:, 4] = self._finished_grip(canonical)
            teacher[inference_rows] = selected_teacher

        if use_reference:
            candidate = torch.zeros(
                (obs.shape[0], len(ACTION_ORDER)),
                dtype=torch.float32,
                device=obs.device,
            )
            candidate[:, 4] = self._finished_grip(canonical)
            if bool(inference_rows.any()):
                self.reference_call_count += 1
                candidate[inference_rows] = selected_teacher
            source = "reference"
        else:
            if self.policy_source is None:
                raise RuntimeError(
                    f"no policy action source configured for {canonical.value}"
                )
            candidate = torch.zeros(
                (obs.shape[0], len(ACTION_ORDER)),
                dtype=torch.float32,
                device=obs.device,
            )
            candidate[:, 4] = self._finished_grip(canonical)
            if bool(inference_rows.any()):
                selected = torch.as_tensor(
                    self.policy_source.action(canonical, obs[inference_rows]),
                    device=obs.device,
                    dtype=torch.float32,
                )
                self._validate_action(
                    canonical, selected, int(inference_rows.sum()), "policy"
                )
                candidate[inference_rows] = selected
            source = "policy"

        candidate = candidate.clone()
        command = project_skill_action(canonical, candidate)
        command[finished_mask, :4] = 0.0
        command[finished_mask, 4] = self._finished_grip(canonical)
        return ActionOutput(
            skill=canonical,
            command=command,
            candidate=candidate,
            active=active,
            source=source,
            reference=None if teacher is None else teacher.clone(),
        )

    @staticmethod
    def _select_batch_rows(value: object, mask: object, batch: int) -> object:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) == batch:
            return value[mask]
        return value

    @staticmethod
    def _validate_action(skill: Skill, action: Any, batch: int, label: str) -> None:
        import torch

        expected = (batch, len(ACTION_ORDER))
        if action.shape != expected or not torch.isfinite(action).all():
            raise ValueError(
                f"{skill.value} {label} action must be finite with shape {expected}"
            )

    @staticmethod
    def _finished_grip(skill: Skill) -> float:
        return 1.0 if skill in {
            Skill.REACH,
            Skill.RELEASE_STABILIZE,
            Skill.RETREAT,
        } else -1.0
