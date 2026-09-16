"""Isaac Lab bridge that keeps inference inside the V7 pipeline boundary.

Isaac Lab and the migrated V5 environment use batched torch tensors.  The V7
core deliberately exposes dependency-free, single-observation contracts.  This
adapter is the application-layer seam between those two designs: it dispatches
each batch row through :class:`StageVLAPipeline` and returns one torch tensor to
the simulator wrapper.
"""

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
from stage_vla_v7.contracts import ModelDescriptor, SKILL_SEQUENCE, Skill
from stage_vla_v7.orchestration import PreparedTask, StageVLAPipeline


def build_torchscript_cube_service(
    checkpoints: Mapping[Skill | str, str | Path],
    *,
    device: str = "cpu",
    bundle_name: str = "v7-migrated-v5-cube",
) -> ActionService:
    """Build the exact validated 4 cm/50 g cube action service.

    Args:
        checkpoints: One TorchScript checkpoint for every canonical skill.
        device: Torch device used for inference.
        bundle_name: Stable audit name recorded in action diagnostics.

    Returns:
        A complete action service with fail-closed physical-domain routing.
    """
    normalized = {Skill(skill): Path(path).resolve() for skill, path in checkpoints.items()}
    missing = set(SKILL_SEQUENCE) - set(normalized)
    extra = set(normalized) - set(SKILL_SEQUENCE)
    if missing or extra:
        raise ValueError(
            "checkpoint mapping mismatch: "
            f"missing={sorted(skill.value for skill in missing)}, "
            f"extra={sorted(skill.value for skill in extra)}"
        )
    dimensions = {skill: (52 if skill is Skill.REACH else 55) for skill in SKILL_SEQUENCE}
    policies = {
        skill: TorchScriptActionPolicy(
            normalized[skill],
            skill=skill,
            observation_dim=dimensions[skill],
            device=device,
            version="v7-migrated-v5-cube-v1",
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

    def __init__(self, pipeline: StageVLAPipeline, prepared: PreparedTask) -> None:
        self.pipeline = pipeline
        self.prepared = prepared
        self._relation_index: int | None = None
        self._tokens = {
            (token.relation_index, token.skill): token for token in prepared.tokens
        }
        self._rows_by_skill: Counter[str] = Counter()
        self._rows_by_token: Counter[tuple[int, str]] = Counter()
        self._batch_calls_by_skill: Counter[str] = Counter()
        self._safety_projections = 0
        self._providers: set[str] = set()
        self._bundles: set[str] = set()

    def bind_relation(self, relation_index: int) -> None:
        """Select the prepared relation served by subsequent simulator calls."""
        if relation_index < 0:
            raise ValueError("relation_index must be non-negative")
        if not any(index == relation_index for index, _skill in self._tokens):
            raise LookupError(f"prepared task has no relation {relation_index}")
        self._relation_index = relation_index

    def action(self, skill: object, observation: Any) -> Any:
        """Return one V7-routed action for every row in a torch observation batch."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - Isaac Lab supplies torch
            raise RuntimeError("PipelineActionSource requires torch") from exc

        if self._relation_index is None:
            raise RuntimeError("bind_relation must be called before action inference")
        canonical_skill = Skill(getattr(skill, "value", skill))
        token = self._tokens[(self._relation_index, canonical_skill)]
        batch = torch.as_tensor(observation)
        if batch.ndim != 2 or not torch.isfinite(batch).all():
            raise ValueError("observation must be a finite rank-2 tensor")

        actions: list[tuple[float, float, float, float, float]] = []
        for row in batch.detach().cpu().tolist():
            result = self.pipeline.act(self.prepared, token, row)
            actions.append(result.action.values)
            self._providers.add(result.provider.name)
            bundle = result.diagnostics.get("bundle")
            if isinstance(bundle, str):
                self._bundles.add(bundle)
            self._safety_projections += int(bool(result.diagnostics.get("safety_projected")))

        self._rows_by_skill[canonical_skill.value] += len(actions)
        self._rows_by_token[(self._relation_index, canonical_skill.value)] += len(actions)
        self._batch_calls_by_skill[canonical_skill.value] += 1
        return torch.tensor(actions, dtype=torch.float32, device=batch.device)

    def audit(self) -> dict[str, object]:
        """Return serializable evidence that every learned action used V7."""
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
