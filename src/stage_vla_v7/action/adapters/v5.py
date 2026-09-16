"""Composition helper for the frozen V5 generalized-cube bundle."""

from __future__ import annotations

from pathlib import Path

from stage_vla_v7.contracts import ModelDescriptor, SKILL_SEQUENCE, Skill

from ..domain import ActionBundle, PolicyDomain
from .torchscript import TorchScriptActionPolicy


def build_v5_cube_bundle(root: str | Path, *, device: str = "cpu") -> ActionBundle:
    """Load all eight policies while preserving the exact validated V5 domain."""
    bundle_root = Path(root).resolve()
    dimensions = {skill: (52 if skill is Skill.REACH else 55) for skill in SKILL_SEQUENCE}
    policies = {
        skill: TorchScriptActionPolicy(
            bundle_root / skill.value.lower() / "policy.ts",
            skill=skill,
            observation_dim=dimensions[skill],
            device=device,
            version="v5-generalized-cube-v1",
        )
        for skill in SKILL_SEQUENCE
    }
    domain = PolicyDomain(
        object_geometries=("box",),
        support_geometries=("box",),
        object_size_min_m=(0.04, 0.04, 0.04),
        object_size_max_m=(0.04, 0.04, 0.04),
        support_size_min_m=(0.04, 0.04, 0.04),
        support_size_max_m=(0.04, 0.04, 0.04),
        object_mass_range_kg=(0.05, 0.05),
        support_mass_range_kg=(0.05, 0.05),
    )
    return ActionBundle(
        ModelDescriptor(
            "v5-generalized-cube",
            "1",
            "action-bundle",
            ("eight-skill", "oracle-state", "rigid-cube"),
        ),
        domain,
        policies,
    )
