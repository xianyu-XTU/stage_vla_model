"""Pure whole-chain audit gates for physical evaluation."""

from __future__ import annotations

from typing import Mapping, Sequence


def verify_v7_chain(
    *,
    use_vision: bool,
    learned_reach: bool,
    reference_skills: Sequence[str],
    reach_reference_recovery_used: bool,
    reference_skill_calls: int,
    recovery_calls: int,
    vision: Mapping[str, object],
    pipeline_audit: Mapping[str, object],
    runtime_purity: Mapping[str, object],
) -> bool:
    """Require a complete learned chain with strict RGB-D and a pure V7 runtime."""
    try:
        loaded_v5_module_count = int(
            runtime_purity.get("loaded_v5_module_count", 1)
        )
    except (TypeError, ValueError):
        return False
    return bool(
        use_vision
        and learned_reach
        and not reference_skills
        and not reach_reference_recovery_used
        and int(reference_skill_calls) == 0
        and int(recovery_calls) == 0
        and vision.get("strict_mode") is True
        and int(vision.get("v7_service_calls", 0)) > 0
        and int(vision.get("invalid_frames", 0)) == 0
        and int(vision.get("oracle_fallback_count", 0)) == 0
        and pipeline_audit.get("all_prepared_skills_exercised") is True
        and runtime_purity.get("vendor_path_exposed") is False
        and loaded_v5_module_count == 0
        and runtime_purity.get("verified") is True
        and runtime_purity.get("import_blocker_enabled") is True
    )


__all__ = ["verify_v7_chain"]
