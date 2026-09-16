"""Static capability checks for compiled stack chains.

This module deliberately does not claim physical success.  It reports whether
the current frozen v5 action-model route can address every relation in a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import SkillToken
from ..objects import ObjectDomain, ObjectModule, rigid_cube_domain


@dataclass(frozen=True)
class ChainCapabilityReport:
    planned: bool
    action_models_cover_all_stages: bool
    can_complete_with_current_route: bool
    unsupported_relations: tuple[tuple[str, str], ...]
    blockers: tuple[str, ...]


def assess_chain_capability(
    tokens: Iterable[SkillToken],
    *,
    object_domain: ObjectDomain | None = None,
    supported_objects: tuple[str, ...] | None = None,
    validated_relation_chains: tuple[tuple[tuple[str, str], ...], ...] = (),
) -> ChainCapabilityReport:
    """Check a compiled chain against a color-independent physical domain.

    ``supported_objects`` remains as an optional compatibility restriction for
    old reports. New callers should provide ``object_domain`` when selecting a
    non-cube action bundle.
    """
    token_list = tuple(tokens)
    relations: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in token_list:
        relation = (token.object_name, token.target_name)
        if relation not in seen:
            seen.add(relation)
            relations.append(relation)

    domain = object_domain or rigid_cube_domain()
    domain.validate()
    supported_labels = None if supported_objects is None else set(supported_objects)
    unsupported_list: list[tuple[str, str]] = []
    for relation in relations:
        if supported_labels is not None and not set(relation).issubset(supported_labels):
            unsupported_list.append(relation)
            continue
        try:
            module = ObjectModule(*relation)
        except ValueError:
            unsupported_list.append(relation)
            continue
        if not module.supports_operation("stack") or not domain.supports(module):
            unsupported_list.append(relation)
    unsupported = tuple(unsupported_list)
    blockers: list[str] = []
    if not token_list:
        blockers.append("the plan compiled to no action tokens")
    if unsupported:
        blockers.append(
            "the declared action-model physical domain does not include every relation; "
            "a compatible action bundle or expanded domain training is required"
        )
    covered = bool(token_list) and not unsupported
    route_validated = tuple(relations) in set(validated_relation_chains)
    if not route_validated:
        blockers.append(
            "no continuous no-reset rollout has been validated for this exact relation chain"
        )
    return ChainCapabilityReport(
        planned=bool(token_list),
        action_models_cover_all_stages=covered,
        can_complete_with_current_route=covered and route_validated,
        unsupported_relations=unsupported,
        blockers=tuple(blockers),
    )
