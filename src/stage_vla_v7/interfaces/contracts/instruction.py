"""Language-neutral instruction contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ContractError


@dataclass(frozen=True)
class StackRelation:
    """One semantic instruction to place an object on a support object."""

    object_label: str
    support_label: str

    def __post_init__(self) -> None:
        for name in ("object_label", "support_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} must be non-empty")
        if self.object_label == self.support_label:
            raise ContractError("an object cannot be stacked on itself")
