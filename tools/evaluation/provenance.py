"""Collect reproducibility metadata for physical-evaluation evidence."""

from __future__ import annotations

from dataclasses import dataclass
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
    return completed.stdout.rstrip()


@dataclass(frozen=True)
class SourceSnapshot:
    """Git identity captured before an evaluator or evidence tool starts work."""

    commit: str
    worktree_clean: bool
    git_status: tuple[str, ...]
    captured_at_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_commit": self.commit,
            "source_worktree_clean_before_run": self.worktree_clean,
            "source_clean_before_run": self.worktree_clean,
            "source_git_status_before_run": list(self.git_status),
            "source_snapshot_at_utc": self.captured_at_utc,
        }


def capture_source_snapshot(repository_root: Path) -> SourceSnapshot:
    """Capture source provenance before runtime outputs can dirty the tree."""
    root = Path(repository_root).resolve()
    status_lines = tuple(
        line for line in _git(root, "status", "--porcelain=v1").splitlines() if line
    )
    return SourceSnapshot(
        commit=_git(root, "rev-parse", "HEAD"),
        worktree_clean=not status_lines,
        git_status=status_lines,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def collect_evidence_provenance(
    plan: "EvaluationPlan",
    *,
    repository_root: Path,
    source_snapshot: SourceSnapshot | None = None,
) -> dict[str, object]:
    """Record code, lock, and actual checkpoint identities for one result."""
    root = Path(repository_root).resolve()
    source = source_snapshot or capture_source_snapshot(root)
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
        **source.as_dict(),
        "git_commit": commit,
        "git_worktree_dirty": bool(status_lines),
        "git_status": list(status_lines),
        "artifact_lock": lock,
        "checkpoint_hashes": checkpoint_hashes,
    }


__all__ = [
    "SourceSnapshot",
    "capture_source_snapshot",
    "collect_evidence_provenance",
]
