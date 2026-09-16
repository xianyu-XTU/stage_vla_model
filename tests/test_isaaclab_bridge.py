from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stage_vla_v7.contracts import ObjectDetection, SceneState
from stage_vla_v7.language import DeterministicLanguageProvider, LanguageService
from stage_vla_v7.orchestration import StageVLAPipeline, default_cube_catalog
from stage_vla_v7.simulation.isaac_lab import PipelineActionSource
from stage_vla_v7.vision import StaticVisionProvider, VisionRequest, VisionService


def test_batched_isaaclab_source_routes_every_row_through_pipeline(
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
        "stack the red cube on the blue cube",
        VisionRequest(rgb=object()),
    )
    source = PipelineActionSource(pipeline, prepared)
    source.bind_relation(0)

    actions = source.action("REACH", torch.zeros((2, 3)))

    assert actions.shape == (2, 5)
    assert actions[0].tolist() == pytest.approx([1.0, -1.0, 0.25, 1.0, -0.5])
    audit = source.audit()
    assert audit["inference_rows_by_skill"] == {"REACH": 2}
    assert audit["action_bundles"] == ["test-cube-bundle"]


def test_legacy_integration_path_reexports_canonical_bridge() -> None:
    from stage_vla_v7.integrations import PipelineActionSource as LegacySource

    assert LegacySource is PipelineActionSource
