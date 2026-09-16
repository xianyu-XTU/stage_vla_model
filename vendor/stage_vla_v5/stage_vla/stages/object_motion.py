"""Per-object Cartesian motion limits shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RigidObjectMotionProfile:
    """Maximum Cartesian translation per control step for each moving skill."""

    lift_translation_limit_m: float = 0.005
    transport_translation_limit_m: float = 0.005
    align_translation_limit_m: float = 0.005
    descend_translation_limit_m: float = 0.003
    release_translation_limit_m: float = 0.003
    retreat_translation_limit_m: float = 0.005

    def validate(self) -> None:
        values = (
            self.lift_translation_limit_m,
            self.transport_translation_limit_m,
            self.align_translation_limit_m,
            self.descend_translation_limit_m,
            self.release_translation_limit_m,
            self.retreat_translation_limit_m,
        )
        if any(not 0.0 < float(value) <= 0.01 for value in values):
            raise ValueError("motion translation limits must lie in (0, 0.01]")

    def for_skill(self, skill: str) -> float:
        name = "release" if str(skill) == "RELEASE_STABILIZE" else str(skill).lower()
        key = f"{name}_translation_limit_m"
        if not hasattr(self, key):
            raise ValueError(f"skill has no motion limit: {skill!r}")
        return float(getattr(self, key))


def motion_profile_from_config(path: str | Path) -> RigidObjectMotionProfile:
    """Load an optional ``motion`` block while preserving legacy defaults."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("motion", {})
    defaults = RigidObjectMotionProfile()
    profile = RigidObjectMotionProfile(**{
        field: float(values.get(field, getattr(defaults, field)))
        for field in defaults.__dataclass_fields__
    })
    profile.validate()
    return profile
