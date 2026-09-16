"""Provider identity contract."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ContractError


@dataclass(frozen=True)
class ModelDescriptor:
    """Stable identity and advertised capabilities of one provider."""

    name: str
    version: str
    kind: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "kind"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"descriptor {field_name} must be non-empty")
        capabilities = tuple(self.capabilities)
        if len(capabilities) != len(set(capabilities)):
            raise ContractError("descriptor capabilities must not contain duplicates")
        object.__setattr__(self, "capabilities", capabilities)
