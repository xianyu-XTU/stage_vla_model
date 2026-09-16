from __future__ import annotations

import ast
from pathlib import Path

import torch

from tools.evaluation.task_evaluator import evaluate_task


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "tools" / "evaluation"
KNOWN_SIZE_ENVIRONMENT = (
    ROOT
    / "src"
    / "stage_vla_v7"
    / "simulation"
    / "isaac_lab"
    / "known_size_environment.py"
)


def _assigned_env_attribute(target: ast.expr) -> str | None:
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "env"
    ):
        return target.attr
    if isinstance(target, ast.Subscript):
        return _assigned_env_attribute(target.value)
    return None


def test_compatibility_entry_only_forwards_to_cli() -> None:
    entry = ROOT / "tools" / "eval_v7_multicube_chain.py"
    source = entry.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "tools.evaluation.cli"
        for alias in node.names
    }
    assert imported == {"main"}
    assert len(source.splitlines()) <= 25


def test_evaluator_modules_remain_bounded_by_responsibility() -> None:
    line_counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in EVALUATION_ROOT.glob("*.py")
    }
    assert max(line_counts.values()) <= 700, line_counts
    assert line_counts["episode_runner.py"] < line_counts["cli.py"]
    assert line_counts["task_evaluator.py"] < 200
    assert line_counts["result_writer.py"] < 300


def test_evaluator_uses_public_environment_boundary() -> None:
    forbidden_mutations = {
        "arm_locked",
        "auto_reset",
        "entry_red_z",
        "episode_steps",
        "finished",
        "last_failure",
        "last_success",
        "last_timeout",
        "max_episode_length",
        "measured",
        "physical_cfg",
        "prev_force",
        "skill",
        "stable_count",
        "stable_steps",
        "steps",
    }
    private_accesses: list[str] = []
    mutations: list[str] = []
    for path in EVALUATION_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "env"
                and node.attr.startswith("_")
            ):
                private_accesses.append(f"{path.name}:{node.lineno}:{node.attr}")
            targets: list[ast.expr] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                )
            for target in targets:
                attribute = _assigned_env_attribute(target)
                if attribute in forbidden_mutations:
                    mutations.append(f"{path.name}:{node.lineno}:{attribute}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.endswith("_")
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "env"
                and node.func.value.attr in forbidden_mutations
            ):
                mutations.append(
                    f"{path.name}:{node.lineno}:{node.func.value.attr}.{node.func.attr}"
                )

    assert not private_accesses, private_accesses
    assert not mutations, mutations


def test_known_size_environment_declares_public_evaluation_api() -> None:
    tree = ast.parse(KNOWN_SIZE_ENVIRONMENT.read_text(encoding="utf-8"))
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnownSizeGraspEnvironment"
    )
    methods = {
        node.name for node in environment.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "configure_evaluation",
        "configure_skill",
        "observe",
        "synchronize_after_external_step",
    } <= methods


def test_single_relation_aggregation_preserves_existing_success() -> None:
    alive = torch.tensor([True, False], dtype=torch.bool)
    result = evaluate_task(
        args=object(),
        raw=object(),
        env=object(),
        scene_assets=("cube_1", "cube_2"),
        task_pairs=(("cube_2", "cube_1"),),
        relation_results=({"passed": True},),
        reset_seen=False,
        overall_alive=alive,
        validation_envs=(0,),
        capture_video_frame=lambda _stage, _step: None,
    )

    assert result.passed is True
    assert result.final_stack is None
    assert result.overall_alive is alive
