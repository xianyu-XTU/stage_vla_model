"""Persist successful demonstrations and DAgger labels from an evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from stage_vla_v7.simulation.physics import (
    ACTION_CONTEXT_ORDER,
    ACTION_CONTEXT_VERSION,
)


SKILLS = (
    "REACH",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "ALIGN",
    "DESCEND",
    "RELEASE_STABILIZE",
    "RETREAT",
)


class SkillDemonstrationBuffer:
    """CPU trajectory buffer for optional demonstration and DAgger output."""

    def __init__(self, skills: Iterable[str] = SKILLS):
        names = tuple(str(skill).upper() for skill in skills)
        unknown = sorted(set(names) - set(SKILLS))
        if unknown:
            raise ValueError(f"unsupported skills: {unknown}")
        self._rows = {
            skill: {"observations": [], "actions": []} for skill in names
        }

    def append(
        self,
        skill: str,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        skill = str(skill).upper()
        if skill not in self._rows:
            raise ValueError(f"skill is not enabled for collection: {skill}")
        observations = torch.as_tensor(observations).detach().cpu().float()
        actions = torch.as_tensor(actions).detach().cpu().float()
        if observations.ndim != 2 or actions.shape != (len(observations), 5):
            raise ValueError(
                "demonstrations require [N,D] observations and [N,5] actions"
            )
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise ValueError("demonstrations contain non-finite values")
        if len(observations):
            self._rows[skill]["observations"].append(observations)
            self._rows[skill]["actions"].append(actions.clamp(-1.0, 1.0))

    def payload(self, skill: str, **metadata) -> dict:
        skill = str(skill).upper()
        rows = self._rows.get(skill)
        if rows is None:
            raise ValueError(f"skill is not enabled for collection: {skill}")
        if not rows["observations"]:
            raise ValueError(f"no demonstrations collected for {skill}")
        observations = torch.cat(rows["observations"], dim=0)
        actions = torch.cat(rows["actions"], dim=0)
        reserved = {
            "skill",
            "observation_dim",
            "action_dim",
            "observations",
            "actions",
        }
        conflict = sorted(reserved.intersection(metadata))
        if conflict:
            raise ValueError(f"metadata cannot override reserved fields: {conflict}")
        contract = _validate_observation_contract(
            int(observations.shape[1]), metadata, f"{skill} payload"
        )
        if contract["object_context_version"] is not None:
            metadata = {
                **metadata,
                "state_observation_dim": contract["state_observation_dim"],
                "object_context_dim": contract["object_context_dim"],
                "object_context_version": contract["object_context_version"],
            }
        return {
            "skill": skill,
            "observation_dim": int(observations.shape[1]),
            "action_dim": 5,
            **metadata,
            "observations": observations,
            "actions": actions,
        }

    def sample_counts(self) -> dict[str, int]:
        return {
            skill: sum(len(batch) for batch in rows["observations"])
            for skill, rows in self._rows.items()
        }


def _validate_observation_contract(
    observation_dim: int,
    metadata: Mapping[str, object],
    source: str,
) -> dict[str, object]:
    version = metadata.get("object_context_version")
    state_value = metadata.get("state_observation_dim")
    context_value = metadata.get("object_context_dim")
    if version is None:
        if state_value is not None and int(state_value) != observation_dim:
            raise ValueError(
                f"legacy state_observation_dim must equal observation_dim in {source}"
            )
        if context_value is not None and int(context_value) != 0:
            raise ValueError(f"legacy object_context_dim must be zero in {source}")
        return {
            "observation_dim": observation_dim,
            "state_observation_dim": observation_dim,
            "object_context_dim": 0,
            "object_context_version": None,
        }
    if version != ACTION_CONTEXT_VERSION:
        raise ValueError(f"unsupported object context version in {source}: {version!r}")
    if state_value is None:
        raise ValueError(f"state_observation_dim is required in {source}")
    state_dim = int(state_value)
    context_dim = len(ACTION_CONTEXT_ORDER)
    if context_value is not None and int(context_value) != context_dim:
        raise ValueError(f"object_context_dim must be {context_dim} in {source}")
    if state_dim < 1 or state_dim + context_dim != observation_dim:
        raise ValueError(
            f"observation_dim must equal state_observation_dim + {context_dim} in {source}"
        )
    return {
        "observation_dim": observation_dim,
        "state_observation_dim": state_dim,
        "object_context_dim": context_dim,
        "object_context_version": version,
    }


@dataclass(frozen=True)
class DataCollectionManifests:
    demonstrations: dict[str, object] | None
    dagger_labels: dict[str, object] | None


def write_data_manifests(
    *,
    demonstration_dir: Path | None,
    dagger_dir: Path | None,
    demonstration_rows: Any,
    dagger_rows: Any,
    passed: bool,
    object_geometry: str,
    grasp_profile: Any,
    output_path: Path,
    reach_checkpoint: Path | None,
    checkpoints: Mapping[str, Path],
    reach_reference_recovery_used: bool,
) -> DataCollectionManifests:
    """Write optional collection payloads without participating in control."""
    import torch

    demonstration_manifest = None
    if demonstration_dir is not None and passed:
        demonstration_manifest = {}
        for skill in demonstration_rows.sample_counts():
            payload = demonstration_rows.payload(
                skill,
                object_geometry=object_geometry,
                grasp_profile={
                    "geometry": grasp_profile.geometry,
                    "height_ratio": grasp_profile.height_ratio,
                    "width_ratio": grasp_profile.width_ratio,
                },
            )
            path = demonstration_dir / f"{skill.lower()}.pt"
            torch.save(payload, path)
            demonstration_manifest[skill] = {
                "path": str(path),
                "samples": int(len(payload["observations"])),
                "observation_dim": int(payload["observation_dim"]),
            }
        (demonstration_dir / "manifest.json").write_text(
            json.dumps({
                "status": "complete",
                "source_result": str(output_path),
                "object_geometry": object_geometry,
                "skills": demonstration_manifest,
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    dagger_manifest = None
    if dagger_dir is not None:
        dagger_manifest = {}
        for skill, count in dagger_rows.sample_counts().items():
            if not count:
                continue
            payload = dagger_rows.payload(
                skill,
                object_geometry=object_geometry,
                collection_mode="model_executed_teacher_labeled",
                source_checkpoint=str(
                    reach_checkpoint if skill == "REACH" else checkpoints[skill]
                ),
                source_result=str(output_path),
            )
            path = dagger_dir / f"{skill.lower()}.pt"
            torch.save(payload, path)
            dagger_manifest[skill] = {
                "path": str(path),
                "samples": count,
                "observation_dim": int(payload["observation_dim"]),
            }
        (dagger_dir / "manifest.json").write_text(
            json.dumps({
                "status": "complete" if dagger_manifest else "empty",
                "collection_mode": "model_executed_teacher_labeled",
                "teacher_controlled_environment": reach_reference_recovery_used,
                "teacher_controlled_labeled_transitions": False,
                "teacher_recovery_used_after_policy_horizon": (
                    reach_reference_recovery_used
                ),
                "source_result": str(output_path),
                "skills": dagger_manifest,
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    return DataCollectionManifests(
        demonstrations=demonstration_manifest,
        dagger_labels=dagger_manifest,
    )


__all__ = [
    "DataCollectionManifests",
    "SkillDemonstrationBuffer",
    "write_data_manifests",
]
