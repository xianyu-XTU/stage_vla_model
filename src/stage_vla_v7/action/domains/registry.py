"""Fail-closed physical-domain router."""

from __future__ import annotations

from typing import Mapping

from stage_vla_v7.interfaces import ObjectProfile, RoutingError

from .action_bundle import ActionBundle


class ActionRouter:
    """Select exactly one complete action bundle by declared physical coverage."""

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
