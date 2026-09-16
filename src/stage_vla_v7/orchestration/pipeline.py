"""Composition layer connecting the three independent model ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from stage_vla_v7.action import ActionRequest, ActionResult, ActionService
from stage_vla_v7.contracts import SkillToken, expand_skill_tokens
from stage_vla_v7.language import LanguageRequest, LanguageResult, LanguageService
from stage_vla_v7.vision import VisionRequest, VisionResult, VisionService

from .catalog import ObjectCatalog


@dataclass(frozen=True)
class PreparedTask:
    """Bound language and vision results ready for skill execution."""

    language: LanguageResult
    vision: VisionResult
    tokens: tuple[SkillToken, ...]

    def __post_init__(self) -> None:
        expected = expand_skill_tokens(self.language.plan)
        if self.tokens != expected:
            raise ValueError("prepared task tokens do not match the language plan")


class StageVLAPipeline:
    """Compose providers while preserving their independent contracts."""

    def __init__(
        self,
        *,
        vision: VisionService,
        language: LanguageService,
        action: ActionService,
        objects: ObjectCatalog,
    ) -> None:
        self.vision = vision
        self.language = language
        self.action = action
        self.objects = objects

    def prepare(self, command: str, frame: VisionRequest) -> PreparedTask:
        """Observe the scene, interpret text, and bind required labels."""
        visual = self.vision.observe(frame)
        labels = tuple(item.label for item in visual.scene.detections)
        semantic = self.language.interpret(LanguageRequest(command, labels))
        for relation in semantic.plan.execution_relations:
            visual.scene.detection(relation.object_label)
            visual.scene.detection(relation.support_label)
            object_profile = self.objects.resolve(relation.object_label)
            support_profile = self.objects.resolve(relation.support_label)
            if not object_profile.stackable or not support_profile.stackable:
                raise ValueError(
                    f"relation is not stackable: {relation.object_label!r} -> "
                    f"{relation.support_label!r}"
                )
        return PreparedTask(semantic, visual, expand_skill_tokens(semantic.plan))

    def act(
        self,
        prepared: PreparedTask,
        token: SkillToken,
        observation: Sequence[float],
        *,
        finished: bool = False,
    ) -> ActionResult:
        """Run one scheduled skill through the sole continuous-action boundary."""
        if token not in prepared.tokens:
            raise ValueError("skill token does not belong to this prepared task")
        request = ActionRequest(
            skill=token.skill,
            observation=tuple(observation),
            object_profile=self.objects.resolve(token.object_label),
            support_profile=self.objects.resolve(token.support_label),
            finished=finished,
            metadata={
                "relation_index": token.relation_index,
                "vision_provider": prepared.vision.provider.name,
                "language_provider": prepared.language.provider.name,
            },
        )
        return self.action.act(request)
