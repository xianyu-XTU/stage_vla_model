"""Persist successful demonstrations and DAgger labels from an evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


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


__all__ = ["DataCollectionManifests", "write_data_manifests"]
