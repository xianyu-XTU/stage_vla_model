from __future__ import annotations

from stage_vla_v7.contracts import ObjectDetection, SceneState, Skill
from stage_vla_v7.language import DeterministicLanguageProvider, LanguageService
from stage_vla_v7.orchestration import StageVLAPipeline, default_cube_catalog
from stage_vla_v7.vision import StaticVisionProvider, VisionRequest, VisionService


def test_three_modules_compose_only_in_pipeline(cube_action_service) -> None:
    scene = SceneState(
        (
            ObjectDetection("red_cube", (0.4, 0.1, 0.02), confidence=1.0),
            ObjectDetection("blue_cube", (0.5, 0.0, 0.02), confidence=1.0),
        )
    )
    pipeline = StageVLAPipeline(
        vision=VisionService(StaticVisionProvider(scene)),
        language=LanguageService(DeterministicLanguageProvider()),
        action=cube_action_service,
        objects=default_cube_catalog(),
    )
    prepared = pipeline.prepare(
        "stack the red cube on the blue cube",
        VisionRequest(rgb=object(), frame_id="frame-1"),
    )
    assert len(prepared.tokens) == 8
    assert prepared.tokens[0].skill is Skill.REACH
    result = pipeline.act(prepared, prepared.tokens[0], (0.0, 0.0, 0.0))
    assert len(result.action.values) == 5
    assert result.diagnostics["bundle"] == "test-cube-bundle"
