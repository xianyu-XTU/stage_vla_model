"""Fail-closed checkpoint-lock validation for physical evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from . import bootstrap as _bootstrap  # noqa: F401
from stage_vla_v7.contracts import SKILL_SEQUENCE, Skill


def load_action_checkpoint_hashes(path: Path) -> dict[Skill, str]:
    """Load and validate all eight Skill records from the artifact lock."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage_vla_v7.external_artifacts.v2":
        raise ValueError("unsupported artifact lock schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifact lock must contain an artifacts object")
    result: dict[Skill, str] = {}
    for skill in SKILL_SEQUENCE:
        record = artifacts.get(skill.value.lower())
        dimension = 52 if skill is Skill.REACH else 55
        if (
            not isinstance(record, dict)
            or record.get("skill") != skill.value
            or record.get("model_type") != "torchscript_parameter_policy"
            or record.get("observation_dim") != dimension
            or record.get("action_dim") != 5
        ):
            raise ValueError(f"invalid artifact lock record for {skill.value}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid checkpoint SHA256 for {skill.value}")
        result[skill] = digest
    return result


__all__ = ["load_action_checkpoint_hashes"]
