"""Collect reproducibility metadata for physical-evaluation evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

from stage_vla_v7.interfaces import Skill

if TYPE_CHECKING:
    from .cli import EvaluationPlan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def collect_evidence_provenance(
    plan: "EvaluationPlan",
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Record code, lock, and actual checkpoint identities for one result."""
    root = Path(repository_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    status_lines = tuple(
        line for line in _git(root, "status", "--porcelain=v1").splitlines() if line
    )
    checkpoint_paths = {
        Skill.REACH: plan.reach_checkpoint,
        **{Skill(name): path for name, path in plan.checkpoints.items()},
    }
    checkpoint_hashes = {
        skill.value: {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for skill, path in checkpoint_paths.items()
        if path is not None
    }
    lock = None
    if plan.artifact_lock is not None:
        lock = {
            "path": str(plan.artifact_lock),
            "sha256": _sha256(plan.artifact_lock),
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": bool(status_lines),
        "git_status": list(status_lines),
        "artifact_lock": lock,
        "checkpoint_hashes": checkpoint_hashes,
    }


__all__ = ["collect_evidence_provenance"]
