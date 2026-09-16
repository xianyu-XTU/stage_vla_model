"""Freeze the public V7 surface before structural migration."""

from __future__ import annotations

from stage_vla_v7.action import ActionRequest, ActionResult, ActionService
from stage_vla_v7.contracts import RobotAction, SKILL_SEQUENCE, Skill
from stage_vla_v7.integrations import PipelineActionSource
from stage_vla_v7.language import DeterministicLanguageProvider, LanguageRequest
from stage_vla_v7.orchestration import PreparedTask, StageVLAPipeline
from stage_vla_v7.vision import VisionRequest, VisionResult, VisionService


def test_legacy_public_imports_remain_available() -> None:
    assert ActionRequest is not None
    assert ActionResult is not None
    assert ActionService is not None
    assert PipelineActionSource is not None
    assert PreparedTask is not None
    assert StageVLAPipeline is not None
    assert VisionRequest is not None
    assert VisionResult is not None
    assert VisionService is not None


def test_canonical_skill_order_and_action_shape_are_frozen() -> None:
    assert SKILL_SEQUENCE == (
        Skill.REACH,
        Skill.GRASP,
        Skill.LIFT,
        Skill.TRANSPORT,
        Skill.ALIGN,
        Skill.DESCEND,
        Skill.RELEASE_STABILIZE,
        Skill.RETREAT,
    )
    assert RobotAction.from_values((1, 2, 3, 4, 5)).values == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_chinese_command_keeps_the_frozen_skill_plan() -> None:
    provider = DeterministicLanguageProvider()
    result = provider.interpret(
        LanguageRequest(
            "把红色方块放到蓝色方块上",
            ("red_cube", "blue_cube"),
        )
    )
    relation = result.plan.execution_relations[0]
    assert (relation.object_label, relation.support_label) == ("red_cube", "blue_cube")
