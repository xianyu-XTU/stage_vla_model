from __future__ import annotations

import ast
from pathlib import Path

import torch

from tools.evaluation.task_evaluator import evaluate_task


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "tools" / "evaluation"


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
