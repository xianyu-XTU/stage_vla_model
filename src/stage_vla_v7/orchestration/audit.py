"""Serializable audit records produced at the orchestration boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .prepared_task import PreparedTask


@dataclass(frozen=True)
class PreparedTaskAudit:
    language_provider: str
    vision_provider: str
    relation_count: int
    token_count: int
    skills: tuple[str, ...]

    @classmethod
    def from_prepared(cls, prepared: PreparedTask) -> "PreparedTaskAudit":
        return cls(
            prepared.language.provider.name,
            prepared.vision.provider.name,
            len(prepared.language.plan.execution_relations),
            len(prepared.tokens),
            tuple(token.skill.value for token in prepared.tokens),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "language_provider": self.language_provider,
            "vision_provider": self.vision_provider,
            "relations": self.relation_count,
            "tokens": self.token_count,
            "skills": list(self.skills),
        }
