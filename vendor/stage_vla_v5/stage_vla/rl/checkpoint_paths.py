"""Checkpoint-path helpers that do not depend on the process working directory."""

from __future__ import annotations

from pathlib import Path


def resolve_resume_checkpoint_path(raw_path: str | Path, *, project_root: Path) -> Path:
    """Resolve a checkpoint path deterministically relative to the project root.

    ``isaaclab.bat`` may execute with the IsaacLab repository as the process CWD.
    A user-facing relative checkpoint path such as ``logs/rsl_rl/.../model.pt``
    therefore must not be resolved with bare ``Path(...).resolve()``.
    """

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()
