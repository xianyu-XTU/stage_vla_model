"""Deterministic compiler for the v5 closed task DSL."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import SKILL_SEQUENCE, SkillToken, StackChainSpec, TaskSpec


class UnsupportedCommandError(ValueError):
    """Raised when a command is outside the explicitly supported grammar."""


_STACK_RE = re.compile(
    r"^STACK\(\s*object\s*=\s*([A-Za-z0-9_-]+)\s*,\s*"
    r"target\s*=\s*([A-Za-z0-9_-]+)\s*\)$",
    re.IGNORECASE,
)
_STACK_CHAIN_RE = re.compile(r"^STACK_CHAIN\((.+)\)$", re.IGNORECASE)
_RELATION_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*>\s*([A-Za-z0-9_-]+)\s*$")


@dataclass(frozen=True)
class TaskCompiler:
    """Compile one strict stack command into a fixed skill sequence."""

    def parse(self, command: str) -> TaskSpec:
        if not isinstance(command, str):
            raise UnsupportedCommandError("command must be a string")
        match = _STACK_RE.fullmatch(command.strip())
        if match is None:
            raise UnsupportedCommandError("expected STACK(object=<name>,target=<name>)")
        return TaskSpec(object_name=match.group(1), target_name=match.group(2))

    def compile(self, command: str) -> tuple[SkillToken, ...]:
        chain_match = _STACK_CHAIN_RE.fullmatch(command.strip()) if isinstance(command, str) else None
        if chain_match is not None:
            return self.compile_chain(self.parse_chain(command))
        task = self.parse(command)
        return self.compile_task(task)

    def parse_chain(self, command: str) -> StackChainSpec:
        if not isinstance(command, str):
            raise UnsupportedCommandError("command must be a string")
        match = _STACK_CHAIN_RE.fullmatch(command.strip())
        if match is None:
            raise UnsupportedCommandError("expected STACK_CHAIN(object>target,object>target)")
        relations: list[TaskSpec] = []
        for raw_relation in match.group(1).split(","):
            relation = _RELATION_RE.fullmatch(raw_relation)
            if relation is None:
                raise UnsupportedCommandError("each chain relation must use object>target")
            relations.append(TaskSpec(relation.group(1), relation.group(2)))
        return StackChainSpec(tuple(relations))

    def compile_chain(self, chain: StackChainSpec) -> tuple[SkillToken, ...]:
        if not isinstance(chain, StackChainSpec):
            raise TypeError("chain must be a StackChainSpec")
        return tuple(
            token
            for task in chain.execution_tasks
            for token in self.compile_task(task)
        )

    def compile_task(self, task: TaskSpec) -> tuple[SkillToken, ...]:
        """Compile already-validated semantics from a vision/VLM adapter."""
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        return tuple(SkillToken(skill=skill, object_name=task.object_name, target_name=task.target_name) for skill in SKILL_SEQUENCE)
