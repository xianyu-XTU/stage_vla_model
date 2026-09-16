"""Physical-domain routing for independently trained action bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from stage_vla_v7.contracts import (
    ModelDescriptor,
    ObjectProfile,
    RoutingError,
    SKILL_SEQUENCE,
    Skill,
)

from .interfaces import ActionPolicy


def _inside(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


@dataclass(frozen=True)
class PolicyDomain:
    """Validated physical coverage of one action-model bundle."""

    object_geometries: tuple[str, ...]
    support_geometries: tuple[str, ...]
    object_size_min_m: tuple[float, float, float]
    object_size_max_m: tuple[float, float, float]
    support_size_min_m: tuple[float, float, float]
    support_size_max_m: tuple[float, float, float]
    object_mass_range_kg: tuple[float, float]
    support_mass_range_kg: tuple[float, float]
    object_grasp_modes: tuple[str, ...] = ("parallel_jaw",)
    support_modes: tuple[str, ...] = ("flat",)
    allow_deformable: bool = False

    def __post_init__(self) -> None:
        ranges = (
            (self.object_size_min_m, self.object_size_max_m),
            (self.support_size_min_m, self.support_size_max_m),
        )
        for lower, upper in ranges:
            if len(lower) != 3 or len(upper) != 3 or any(a <= 0 or a > b for a, b in zip(lower, upper)):
                raise ValueError("invalid size domain")
        for lower, upper in (self.object_mass_range_kg, self.support_mass_range_kg):
            if lower <= 0 or lower > upper:
                raise ValueError("invalid mass domain")

    @staticmethod
    def _size_supported(
        size: tuple[float, float, float],
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
    ) -> bool:
        return all(a <= value <= b for value, a, b in zip(size, lower, upper))

    def supports(self, object_profile: ObjectProfile, support_profile: ObjectProfile) -> bool:
        """Return whether both semantic roles are inside the trained domain."""
        return all(
            (
                object_profile.geometry in self.object_geometries,
                support_profile.geometry in self.support_geometries,
                object_profile.grasp_mode in self.object_grasp_modes,
                support_profile.support_mode in self.support_modes,
                self._size_supported(
                    object_profile.size_xyz_m,
                    self.object_size_min_m,
                    self.object_size_max_m,
                ),
                self._size_supported(
                    support_profile.size_xyz_m,
                    self.support_size_min_m,
                    self.support_size_max_m,
                ),
                _inside(object_profile.mass_kg, self.object_mass_range_kg),
                _inside(support_profile.mass_kg, self.support_mass_range_kg),
                self.allow_deformable or not object_profile.deformable,
                object_profile.stackable,
                support_profile.stackable,
            )
        )


@dataclass(frozen=True)
class ActionBundle:
    """One complete set of skill policies sharing a physical domain."""

    descriptor: ModelDescriptor
    domain: PolicyDomain
    policies: Mapping[Skill, ActionPolicy]

    def __post_init__(self) -> None:
        missing = set(SKILL_SEQUENCE) - set(self.policies)
        extra = set(self.policies) - set(SKILL_SEQUENCE)
        if missing or extra:
            raise ValueError(
                f"action bundle policy mismatch: missing={sorted(x.value for x in missing)}, "
                f"extra={sorted(str(x) for x in extra)}"
            )

    def policy(self, skill: Skill) -> ActionPolicy:
        return self.policies[skill]


class ActionRouter:
    """Select exactly one bundle by declared physical coverage."""

    def __init__(self, bundles: Mapping[str, ActionBundle]) -> None:
        if not bundles:
            raise ValueError("at least one action bundle is required")
        self.bundles = dict(bundles)

    def route(self, object_profile: ObjectProfile, support_profile: ObjectProfile) -> ActionBundle:
        matches = [
            (name, bundle)
            for name, bundle in self.bundles.items()
            if bundle.domain.supports(object_profile, support_profile)
        ]
        if not matches:
            raise RoutingError(
                f"no action bundle covers {object_profile.label!r} -> {support_profile.label!r}"
            )
        if len(matches) > 1:
            raise RoutingError(
                "ambiguous action bundle coverage: " + ", ".join(name for name, _ in matches)
            )
        return matches[0][1]
