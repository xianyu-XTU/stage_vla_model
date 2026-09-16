"""Dependency-free V7 contract demonstration."""

from __future__ import annotations

import argparse
import json

from .action import (
    ActionBundle,
    ActionRouter,
    ActionService,
    ConstantActionPolicy,
    PolicyDomain,
)
from .contracts import ModelDescriptor, ObjectDetection, RobotAction, SKILL_SEQUENCE, SceneState
from .language import DeterministicLanguageProvider, LanguageService
from .orchestration import StageVLAPipeline, default_cube_catalog
from .vision import StaticVisionProvider, VisionRequest, VisionService


def _demo_pipeline() -> StageVLAPipeline:
    scene = SceneState(
        (
            ObjectDetection("red_cube", (0.42, 0.10, 0.02), confidence=1.0),
            ObjectDetection("blue_cube", (0.55, -0.05, 0.02), confidence=1.0),
        )
    )
    policies = {
        skill: ConstantActionPolicy(
            RobotAction(0.0, 0.0, 0.0, 0.0, 1.0 if skill.value == "REACH" else -1.0),
            observation_dim=1,
            name=f"demo-{skill.value.lower()}",
        )
        for skill in SKILL_SEQUENCE
    }
    domain = PolicyDomain(
        object_geometries=("box",),
        support_geometries=("box",),
        object_size_min_m=(0.04, 0.04, 0.04),
        object_size_max_m=(0.04, 0.04, 0.04),
        support_size_min_m=(0.04, 0.04, 0.04),
        support_size_max_m=(0.04, 0.04, 0.04),
        object_mass_range_kg=(0.05, 0.05),
        support_mass_range_kg=(0.05, 0.05),
    )
    bundle = ActionBundle(
        ModelDescriptor("demo-cube", "1", "action-bundle", ("demo",)),
        domain,
        policies,
    )
    return StageVLAPipeline(
        vision=VisionService(StaticVisionProvider(scene)),
        language=LanguageService(DeterministicLanguageProvider()),
        action=ActionService(ActionRouter({"demo": bundle})),
        objects=default_cube_catalog(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="stack the red cube on the blue cube")
    args = parser.parse_args()
    pipeline = _demo_pipeline()
    prepared = pipeline.prepare(args.command, VisionRequest(rgb=object(), frame_id="demo"))
    payload = {
        "language_provider": prepared.language.provider.name,
        "vision_provider": prepared.vision.provider.name,
        "relations": [
            {
                "object": relation.object_label,
                "support": relation.support_label,
            }
            for relation in prepared.language.plan.execution_relations
        ],
        "skills": [token.skill.value for token in prepared.tokens],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
