"""Deterministic instruction-to-action-chain compilation.

This module is the language seam for the current V5 route.  It intentionally
does not infer object properties and it does not emit continuous actions.  It
only turns a small set of imperative sentences into :class:`TaskSpec` or
:class:`StackChainSpec`, after which the existing skill compiler produces the
canonical REACH-to-RETREAT tokens.

The seam is deliberately replaceable: a future language model can return the
same structured task values without changing the downstream action boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Mapping, Protocol

from .compiler import TaskCompiler, UnsupportedCommandError
from .schema import SkillToken, StackChainSpec, TaskSpec


InstructionTask = TaskSpec | StackChainSpec


class ObjectNameResolver(Protocol):
    """Resolve a user-facing object phrase to a stable scene label.

    The resolver owns naming only.  Size, mass, geometry, friction and grasp
    properties remain in the object catalog/action-domain interfaces.
    """

    def resolve(self, phrase: str) -> str:
        """Return a stable object label or raise ``UnsupportedCommandError``."""


def _key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip().lower()
    value = re.sub(r"[\s_-]+", "_", value)
    return value.strip("_.,!?;:，。！？；：")


def _clean_phrase(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r"^[\"'“”‘’\s]+|[\"'“”‘’\s]+$", "", value)
    value = value.strip(".,!?;:，。！？；：、")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:这个|该|此)\s*", "", value)
    return value.strip()


@dataclass(frozen=True)
class CatalogObjectNameResolver:
    """Small alias resolver backed by labels, with optional project aliases.

    Exact identifiers are accepted even when they are not in the current
    catalog (for example ``cube_5``).  The later object module remains the
    authority for whether that label has physical metadata and an action
    bundle, which keeps this language layer usable for new scenes.
    """

    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        table: dict[str, str] = {}
        # Import lazily so this naming layer remains usable by the task DSL
        # without making object metadata part of the parser ABI.
        try:
            from ..objects import object_labels

            labels = tuple(object_labels())
        except Exception:  # pragma: no cover - catalog import is optional
            labels = ()
        for label in labels:
            table[_key(label)] = label
            table[_key(label.replace("_", " "))] = label

        table.update({
            "红方块": "red_cube",
            "红色方块": "red_cube",
            "红色的方块": "red_cube",
            "红立方体": "red_cube",
            "蓝方块": "blue_cube",
            "蓝色方块": "blue_cube",
            "蓝色的方块": "blue_cube",
            "蓝立方体": "blue_cube",
            "绿方块": "green_cube",
            "绿色方块": "green_cube",
            "绿色的方块": "green_cube",
            "绿立方体": "green_cube",
            "黄方块": "yellow_cube",
            "黄色方块": "yellow_cube",
            "黄色的方块": "yellow_cube",
            "黄立方体": "yellow_cube",
            "白杯": "white_cup",
            "白色杯子": "white_cup",
            "蓝杯": "blue_cup",
            "蓝色杯子": "blue_cup",
        })
        for alias, label in self.aliases.items():
            if not isinstance(alias, str) or not isinstance(label, str):
                raise TypeError("object aliases must map strings to strings")
            if not label or label.strip() != label:
                raise ValueError("object alias targets must be trimmed labels")
            normalized = _key(alias)
            if not normalized:
                raise ValueError("object aliases cannot contain an empty key")
            existing = table.get(normalized)
            if existing is not None and existing != label:
                raise ValueError(f"ambiguous object alias: {alias!r}")
            table[normalized] = label
        object.__setattr__(self, "_table", table)

    def resolve(self, phrase: str) -> str:
        if not isinstance(phrase, str):
            raise UnsupportedCommandError("object phrase must be a string")
        cleaned = _clean_phrase(phrase)
        normalized = _key(cleaned)
        if not normalized:
            raise UnsupportedCommandError("object phrase cannot be empty")
        resolved = self._table.get(normalized)
        if resolved is not None:
            return resolved

        # Useful for generated scenes and future registered object labels.
        ordinal = re.fullmatch(r"第?\s*(\d+)\s*(?:个)?\s*(?:方块|立方体)", cleaned)
        if ordinal:
            return f"cube_{int(ordinal.group(1))}"
        english_cube = re.fullmatch(r"(?:cube|block)[_\s-]*(\d+)", cleaned, re.IGNORECASE)
        if english_cube:
            return f"cube_{int(english_cube.group(1))}"
        if re.fullmatch(r"[a-z][a-z0-9_-]*", normalized):
            # Preserve exact identifiers for the object-property provider.
            return normalized
        raise UnsupportedCommandError(
            f"unknown object phrase {phrase!r}; use a registered label or alias"
        )


_EN_ACTION_RE = re.compile(
    r"^(?:please\s+)?(?:(?:first|then|next)\s+)?"
    r"(?:stack|place|put|move|set|position|lay)\s+"
    r"(?:the\s+)?(?P<object>.+?)\s+"
    r"(?:on\s+top\s+of|on|onto|over|above|to)\s+"
    r"(?:the\s+)?(?P<target>.+?)\s*$",
    re.IGNORECASE,
)
_EN_BARE_RE = re.compile(
    r"^(?:the\s+)?(?P<object>.+?)\s+"
    r"(?:on\s+top\s+of|on|onto|over|above)\s+"
    r"(?:the\s+)?(?P<target>.+?)\s*$",
    re.IGNORECASE,
)
_CN_ACTION_RE = re.compile(
    r"^(?:请\s*)?(?:(?:先|然后|接着|再)\s*)?(?:把|将)?\s*"
    r"(?P<object>.+?)\s*"
    r"(?:叠放到|叠放在|堆放到|堆放在|堆叠到|堆叠在|放置到|放置在|"
    r"摆放到|摆放在|移动到|移到|放到|放在|置于|叠在|堆在|叠放|堆放|堆叠)"
    r"\s*(?P<target>.+?)\s*$"
)
_BARE_RELATION_RE = re.compile(
    r"^(?P<object>[A-Za-z0-9_-]+)\s*>\s*(?P<target>[A-Za-z0-9_-]+)$"
)
_TARGET_SUFFIX_RE = re.compile(
    r"(?:\s*(?:上面|顶部|顶上|之上|上|上方|上边|上面放置))$"
)


def _split_clauses(command: str) -> tuple[str, ...]:
    """Split a deterministic multi-relation instruction into clauses."""
    text = unicodedata.normalize("NFKC", command).strip()
    # Punctuation is unambiguous because object names cannot contain it.
    text = re.sub(r"[,，;；]+", "|", text)
    # Chinese sequencing words are accepted without requiring spaces.
    text = re.sub(r"(?:然后|接着|随后|之后|并且|再)\s*", "|", text)
    return tuple(part.strip() for part in text.split("|") if part.strip())


@dataclass(frozen=True)
class InstructionPlan:
    """Parsed language semantics and the scheduler tokens they produce."""

    command: str
    task: InstructionTask
    tokens: tuple[SkillToken, ...]

    @property
    def is_chain(self) -> bool:
        return isinstance(self.task, StackChainSpec)

    @property
    def execution_tasks(self) -> tuple[TaskSpec, ...]:
        if isinstance(self.task, StackChainSpec):
            return self.task.execution_tasks
        return (self.task,)


@dataclass(frozen=True)
class InstructionParser:
    """Parse instruction-like commands without a language model."""

    resolver: ObjectNameResolver = field(default_factory=CatalogObjectNameResolver)
    compiler: TaskCompiler = field(default_factory=TaskCompiler)

    def parse(self, command: str) -> InstructionTask:
        if not isinstance(command, str) or not command.strip():
            raise UnsupportedCommandError("instruction must be a non-empty string")

        # Preserve the exact closed DSL as a stable low-level escape hatch.
        try:
            return self.compiler.parse(command)
        except UnsupportedCommandError:
            pass
        try:
            return self.compiler.parse_chain(command)
        except UnsupportedCommandError:
            pass

        clauses = _split_clauses(command)
        relations = tuple(self._parse_relation(clause) for clause in clauses)
        if len(relations) == 1:
            return relations[0]
        return StackChainSpec(relations)

    def compile(self, command: str) -> tuple[SkillToken, ...]:
        parsed = self.parse(command)
        if isinstance(parsed, StackChainSpec):
            return self.compiler.compile_chain(parsed)
        return self.compiler.compile_task(parsed)

    def plan(self, command: str) -> InstructionPlan:
        parsed = self.parse(command)
        tokens = (
            self.compiler.compile_chain(parsed)
            if isinstance(parsed, StackChainSpec)
            else self.compiler.compile_task(parsed)
        )
        return InstructionPlan(command=command, task=parsed, tokens=tokens)

    def _parse_relation(self, clause: str) -> TaskSpec:
        value = clause.strip()
        bare = _BARE_RELATION_RE.fullmatch(value)
        if bare is not None:
            return TaskSpec(
                self.resolver.resolve(bare.group("object")),
                self.resolver.resolve(bare.group("target")),
            )

        match = _EN_ACTION_RE.fullmatch(value) or _EN_BARE_RE.fullmatch(value)
        if match is None:
            match = _CN_ACTION_RE.fullmatch(value)
        if match is None:
            raise UnsupportedCommandError(
                "expected STACK(...), object>target, "
                "'stack A on B', or '把A放到B上'"
            )
        target = _TARGET_SUFFIX_RE.sub("", match.group("target")).strip()
        return TaskSpec(
            self.resolver.resolve(match.group("object")),
            self.resolver.resolve(target),
        )


__all__ = [
    "CatalogObjectNameResolver",
    "InstructionParser",
    "InstructionPlan",
    "InstructionTask",
    "ObjectNameResolver",
]
