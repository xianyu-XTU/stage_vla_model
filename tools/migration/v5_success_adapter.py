"""Regression-only adapter for the frozen V5 vectorized success oracle."""

from __future__ import annotations

from pathlib import Path

from .v5_bootstrap import with_v5_reference_path


def legacy_vectorized_skill_success(
    skill: object,
    state: object,
    *,
    v5_root: str | Path | None = None,
    **kwargs: object,
) -> object:
    """Call the V5 oracle only inside an explicit reference-path context."""
    with with_v5_reference_path(v5_root):
        from stage_vla.rl.v5_skill_contracts import skill_success

        return skill_success(skill, state, **kwargs)


__all__ = ["legacy_vectorized_skill_success"]
