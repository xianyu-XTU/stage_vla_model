"""Audited batched action source connecting Isaac Lab to StageVLAPipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stage_vla_v7.action import (
    ActionBundle,
    ActionRouter,
    ActionService,
    PolicyDomain,
    TorchScriptActionPolicy,
)
from stage_vla_v7.interfaces import ModelDescriptor, SKILL_SEQUENCE, Skill
from stage_vla_v7.orchestration import PreparedTask, StageVLAPipeline

from .action_adapter import IsaacActionAdapter
from .observation_adapter import IsaacObservationAdapter


def build_torchscript_cube_service(
    checkpoints: Mapping[Skill | str, str | Path],
    *,
    device: str = "cpu",
    bundle_name: str = "v7-migrated-v5-cube",
    expected_hashes: Mapping[Skill | str, str] | None = None,
) -> ActionService:
    """Build the exact validated 4 cm/50 g cube action service."""
    normalized = {Skill(skill): Path(path).resolve() for skill, path in checkpoints.items()}
    missing = set(SKILL_SEQUENCE) - set(normalized)
    extra = set(normalized) - set(SKILL_SEQUENCE)
    if missing or extra:
        raise ValueError(
            "checkpoint mapping mismatch: "
            f"missing={sorted(skill.value for skill in missing)}, "
            f"extra={sorted(skill.value for skill in extra)}"
        )
    hashes = {Skill(skill): digest for skill, digest in (expected_hashes or {}).items()}
    if expected_hashes is not None and set(hashes) != set(SKILL_SEQUENCE):
        raise ValueError("expected checkpoint hashes must cover every canonical skill")
    dimensions = {skill: (52 if skill is Skill.REACH else 55) for skill in SKILL_SEQUENCE}
    policies = {
        skill: TorchScriptActionPolicy(
            normalized[skill],
            skill=skill,
            observation_dim=dimensions[skill],
            device=device,
            version="v7-migrated-v5-cube-v1",
            expected_sha256=hashes.get(skill),
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
        ModelDescriptor(
            bundle_name,
            "1",
            "action-bundle",
            ("eight-skill", "torchscript", "isaaclab", "rigid-cube"),
        ),
        domain,
        policies,
    )
    return ActionService(ActionRouter({bundle_name: bundle}))


class PipelineActionSource:
    """Expose a prepared V7 pipeline as a V5/Isaac Lab batched action source."""

    def __init__(
        self,
        pipeline: StageVLAPipeline,
        prepared: PreparedTask,
        *,
        observations: IsaacObservationAdapter | None = None,
        actions: IsaacActionAdapter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.prepared = prepared
        self.observations = observations or IsaacObservationAdapter()
        self.actions = actions or IsaacActionAdapter()
        self._relation_index: int | None = None
        self._tokens = {(token.relation_index, token.skill): token for token in prepared.tokens}
        self._rows_by_skill: Counter[str] = Counter()
        self._rows_by_token: Counter[tuple[int, str]] = Counter()
        self._batch_calls_by_skill: Counter[str] = Counter()
        self._safety_projections = 0
        self._providers: set[str] = set()
        self._bundles: set[str] = set()

    def bind_relation(self, relation_index: int) -> None:
        if relation_index < 0:
            raise ValueError("relation_index must be non-negative")
        if not any(index == relation_index for index, _skill in self._tokens):
            raise LookupError(f"prepared task has no relation {relation_index}")
        self._relation_index = relation_index

    def action(self, skill: object, observation: Any) -> Any:
        if self._relation_index is None:
            raise RuntimeError("bind_relation must be called before action inference")
        canonical_skill = Skill(getattr(skill, "value", skill))
        token = self._tokens[(self._relation_index, canonical_skill)]
        rows = self.observations.batch(observation)

        robot_actions = []
        for row in rows:
            result = self.pipeline.act(self.prepared, token, row)
            robot_actions.append(result.action)
            self._providers.add(result.provider.name)
            bundle = result.diagnostics.get("bundle")
            if isinstance(bundle, str):
                self._bundles.add(bundle)
            self._safety_projections += int(bool(result.diagnostics.get("safety_projected")))

        count = len(robot_actions)
        self._rows_by_skill[canonical_skill.value] += count
        self._rows_by_token[(self._relation_index, canonical_skill.value)] += count
        self._batch_calls_by_skill[canonical_skill.value] += 1
        return self.actions.to_tensor(robot_actions, like=observation)

    def audit(self) -> dict[str, object]:
        return {
            "relations": len(self.prepared.language.plan.execution_relations),
            "tokens": len(self.prepared.tokens),
            "language_provider": self.prepared.language.provider.name,
            "vision_provider": self.prepared.vision.provider.name,
            "action_providers": sorted(self._providers),
            "action_bundles": sorted(self._bundles),
            "inference_rows_by_skill": dict(sorted(self._rows_by_skill.items())),
            "inference_rows_by_token": {
                f"{index}:{skill}": count
                for (index, skill), count in sorted(self._rows_by_token.items())
            },
            "batch_calls_by_skill": dict(sorted(self._batch_calls_by_skill.items())),
            "safety_projection_count": self._safety_projections,
            "all_prepared_skills_exercised": all(
                self._rows_by_token[(token.relation_index, token.skill.value)] > 0
                for token in self.prepared.tokens
            ),
        }
