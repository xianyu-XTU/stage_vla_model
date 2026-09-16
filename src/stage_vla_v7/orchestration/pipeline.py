"""Composition layer connecting the three independent model ports."""

from __future__ import annotations

from typing import Sequence

from stage_vla_v7.action import ActionRequest, ActionResult, ActionService, TaskScheduler
from stage_vla_v7.interfaces import RobotObservation, SkillToken
from stage_vla_v7.language import LanguageRequest, LanguageResult, LanguageService
from stage_vla_v7.vision import VisionRequest, VisionResult, VisionService

from .catalog import ObjectCatalog
from .execution_context import ExecutionContext
from .prepared_task import PreparedTask


class StageVLAPipeline:
    """Compose providers while preserving their independent contracts."""

    def __init__(
        self,
        *,
        vision: VisionService,
        language: LanguageService,
        action: ActionService,
        objects: ObjectCatalog,
        scheduler: TaskScheduler | None = None,
    ) -> None:
        self.vision = vision
        self.language = language
        self.action = action
        self.objects = objects
        self.scheduler = scheduler or TaskScheduler()

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
        return PreparedTask(semantic, visual, self.scheduler.schedule(semantic.plan))

    def prepare_context(self, context: ExecutionContext) -> PreparedTask:
        return self.prepare(context.command, context.frame)

    def act(
        self,
        prepared: PreparedTask,
        token: SkillToken,
        observation: Sequence[float] | RobotObservation,
        *,
        finished: bool = False,
    ) -> ActionResult:
        """Run one scheduled skill through the sole continuous-action boundary."""
        if token not in prepared.tokens:
            raise ValueError("skill token does not belong to this prepared task")
        values = observation.values if isinstance(observation, RobotObservation) else observation
        request = ActionRequest(
            skill=token.skill,
            observation=tuple(values),
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
