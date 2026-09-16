from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stage_vla_v7.contracts import ObjectDetection, SceneState, SKILL_SEQUENCE
from stage_vla_v7.language import DeterministicLanguageProvider, LanguageService
from stage_vla_v7.orchestration import StageVLAPipeline, default_cube_catalog
from stage_vla_v7.simulation.isaac_lab import IsaacActionAdapter, PipelineActionSource
from stage_vla_v7.vision import StaticVisionProvider, VisionRequest, VisionService


def test_chinese_command_routes_all_eight_skills_to_isaac_actions(
    cube_action_service,
) -> None:
    scene = SceneState(
        (
            ObjectDetection("red_cube", (0.4, 0.1, 0.02)),
            ObjectDetection("blue_cube", (0.5, 0.0, 0.02)),
        )
    )
    pipeline = StageVLAPipeline(
        vision=VisionService(StaticVisionProvider(scene)),
        language=LanguageService(DeterministicLanguageProvider()),
        action=cube_action_service,
        objects=default_cube_catalog(),
    )
    prepared = pipeline.prepare(
        "把红色方块放到蓝色方块上",
        VisionRequest(rgb=object(), depth_m=object(), frame_id="isaac-rgbd"),
    )
    source = PipelineActionSource(pipeline, prepared)
    source.bind_relation(0)

    emitted = []
    for token in prepared.tokens:
        batch = source.action(token.skill, torch.zeros((1, 3)))
        emitted.append(batch)

    assert tuple(token.skill for token in prepared.tokens) == SKILL_SEQUENCE
    assert all(batch.shape == (1, 5) for batch in emitted)
    assert all(torch.isfinite(batch).all() for batch in emitted)
    simulation_action = IsaacActionAdapter().to_simulation_action(
        pipeline.act(prepared, prepared.tokens[-1], (0.0, 0.0, 0.0)).action
    )
    assert simulation_action.robot_action.values == pytest.approx(
        (1.0, -1.0, 0.25, 1.0, -0.5)
    )
    audit = source.audit()
    assert audit["all_prepared_skills_exercised"] is True
    assert audit["inference_rows_by_skill"] == {skill.value: 1 for skill in SKILL_SEQUENCE}
