from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.evaluation import episode_runner


def test_isaac_launch_failure_records_provenance_and_purity(tmp_path, monkeypatch) -> None:
    output = tmp_path / "failed.json"
    plan = SimpleNamespace(
        args=SimpleNamespace(require_v7_chain=True),
        scene_assets=(),
        skill_translation_limits={},
        task_pairs=(),
        command_text="test command",
        language_service=None,
        label_to_asset={},
        fixed_asset_xyz={},
        asset_layout_file=None,
        checkpoints={},
        reach_checkpoint=None,
        artifact_lock=None,
        expected_checkpoint_hashes={},
        output_path=output,
        demonstration_dir=None,
        dagger_dir=None,
        video_path=None,
        object_positions=None,
        observer_camera_model=None,
    )
    monkeypatch.setattr(episode_runner, "capture_source_snapshot", lambda root: object())
    monkeypatch.setattr(episode_runner, "install_v5_import_blocker", lambda: None)
    monkeypatch.setattr(
        episode_runner,
        "audit_runtime_purity",
        lambda: SimpleNamespace(as_dict=lambda: {"verified": True}),
    )
    monkeypatch.setattr(
        episode_runner,
        "collect_evidence_provenance",
        lambda *args, **kwargs: {"source_commit": "test-commit"},
    )

    class FailedLauncher:
        def __init__(self, args) -> None:
            raise RuntimeError("Isaac launch failed")

    with pytest.raises(RuntimeError, match="Isaac launch failed"):
        episode_runner.run_evaluation(plan, FailedLauncher)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure"] == {
        "type": "RuntimeError",
        "message": "Isaac launch failed",
    }
    assert result["v7_chain"]["verified"] is False
    assert result["runtime_purity"] == {"verified": True}
    assert result["provenance"] == {"source_commit": "test-commit"}
