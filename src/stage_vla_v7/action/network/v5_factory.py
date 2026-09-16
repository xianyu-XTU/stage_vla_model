"""Factory for the frozen V5-trained, V7-routed cube bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from stage_vla_v7.interfaces import ModelDescriptor, Skill

from ..action_list import ACTION_REGISTRY
from ..domains import ActionBundle, PolicyDomain
from .torchscript_policy import TorchScriptActionPolicy


def build_v5_cube_bundle(
    root: str | Path,
    *,
    device: str = "cpu",
    expected_hashes: Mapping[Skill | str, str] | None = None,
) -> ActionBundle:
    """Load all eight policies without widening their validated physical domain."""
    bundle_root = Path(root).resolve()
    hashes = {Skill(skill): value for skill, value in (expected_hashes or {}).items()}
    policies = {
        definition.skill: TorchScriptActionPolicy(
            bundle_root / definition.skill.value.lower() / "policy.ts",
            skill=definition.skill,
            observation_dim=definition.legacy_observation_dim,
            device=device,
            version="v5-generalized-cube-v1",
            expected_sha256=hashes.get(definition.skill),
        )
        for definition in ACTION_REGISTRY.definitions
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
