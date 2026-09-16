"""Pure whole-chain audit gates for physical evaluation."""

from __future__ import annotations

from typing import Mapping, Sequence


def verify_v7_chain(
    *,
    use_vision: bool,
    learned_reach: bool,
    reference_skills: Sequence[str],
    reach_reference_recovery_used: bool,
    vision: Mapping[str, object],
    pipeline_audit: Mapping[str, object],
) -> bool:
    """Require a complete learned chain with strict, non-oracle RGB-D input."""
    return bool(
        use_vision
        and learned_reach
        and not reference_skills
        and not reach_reference_recovery_used
        and int(vision.get("v7_service_calls", 0)) > 0
        and int(vision.get("invalid_frames", 0)) == 0
        and int(vision.get("oracle_fallback_count", 0)) == 0
        and pipeline_audit.get("all_prepared_skills_exercised") is True
    )


__all__ = ["verify_v7_chain"]
