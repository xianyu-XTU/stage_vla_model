"""Utilities for collecting and combining per-skill action demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch

from ..objects import ACTION_CONTEXT_ORDER, ACTION_CONTEXT_VERSION


SKILLS = (
    "REACH", "GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND",
    "RELEASE_STABILIZE", "RETREAT",
)


class SkillDemonstrationBuffer:
    """CPU trajectory buffer with explicit observation/action contracts."""

    def __init__(self, skills: Iterable[str] = SKILLS):
        names = tuple(str(skill).upper() for skill in skills)
        unknown = sorted(set(names) - set(SKILLS))
        if unknown:
            raise ValueError(f"unsupported skills: {unknown}")
        self._rows = {
            skill: {"observations": [], "actions": []} for skill in names
        }

    def append(self, skill: str, observations: torch.Tensor,
               actions: torch.Tensor) -> None:
        skill = str(skill).upper()
        if skill not in self._rows:
            raise ValueError(f"skill is not enabled for collection: {skill}")
        observations = torch.as_tensor(observations).detach().cpu().float()
        actions = torch.as_tensor(actions).detach().cpu().float()
        if observations.ndim != 2 or actions.shape != (len(observations), 5):
            raise ValueError("demonstrations require [N,D] observations and [N,5] actions")
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
        reserved = {"skill", "observation_dim", "action_dim", "observations", "actions"}
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


def load_combined_skill_dataset(
    skill: str, directories: Iterable[Path]
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Load one skill from multiple compatible demonstration directories."""
    skill = str(skill).upper()
    if skill not in SKILLS:
        raise ValueError(f"unsupported skill: {skill}")
    observation_batches = []
    action_batches = []
    sources = []
    observation_dim = None
    observation_contract = None
    for directory in directories:
        path = Path(directory).resolve() / f"{skill.lower()}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("skill", "")).upper() != skill:
            raise ValueError(f"skill mismatch in {path}")
        observations = torch.as_tensor(payload["observations"]).float()
        actions = torch.as_tensor(payload["actions"]).float()
        current_dim = int(payload["observation_dim"])
        current_contract = _validate_observation_contract(
            current_dim, payload, str(path)
        )
        if observations.ndim != 2 or observations.shape[1] != current_dim:
            raise ValueError(f"invalid observations in {path}")
        if actions.shape != (len(observations), 5):
            raise ValueError(f"invalid actions in {path}")
        if observation_dim is None:
            observation_dim = current_dim
        elif current_dim != observation_dim:
            raise ValueError(f"observation dimension mismatch in {path}")
        if observation_contract is None:
            observation_contract = current_contract
        elif current_contract != observation_contract:
            raise ValueError(f"observation contract mismatch in {path}")
        observation_batches.append(observations)
        action_batches.append(actions.clamp(-1.0, 1.0))
        sources.append({
            "path": str(path),
            "samples": int(len(observations)),
            **current_contract,
        })
    if not observation_batches:
        raise FileNotFoundError(f"no {skill} dataset found in supplied directories")
    return torch.cat(observation_batches), torch.cat(action_batches), sources


def _validate_observation_contract(
    observation_dim: int,
    metadata: Mapping[str, object],
    source: str,
) -> dict[str, object]:
    """Normalize and validate legacy or physical-context observation metadata."""
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
        raise ValueError(
            f"object_context_dim must be {context_dim} in {source}"
        )
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
