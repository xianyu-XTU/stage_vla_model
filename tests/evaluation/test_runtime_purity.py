from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.evaluation.audit import verify_v7_chain
from tools.evaluation.provenance import (
    capture_source_snapshot,
    collect_evidence_provenance,
)
from tools.evaluation.runtime_purity import (
    audit_runtime_purity,
    install_v5_import_blocker,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOTS = (ROOT / "src" / "stage_vla_v7", ROOT / "tools" / "evaluation")


def _is_v5_name(name: str) -> bool:
    return name == "stage_vla" or name.startswith("stage_vla.")


def _isolated(script: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment.pop("STAGE_VLA_V5_ROOT", None)
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _import_probe(*modules: str, block_v5: bool = False) -> dict[str, object]:
    setup = ""
    if block_v5:
        setup = """
class V5ImportBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'stage_vla' or fullname.startswith('stage_vla.'):
            raise ImportError(f'blocked V5 import: {fullname}')
        return None
sys.meta_path.insert(0, V5ImportBlocker())
"""
    imports = "\n".join(f"importlib.import_module({name!r})" for name in modules)
    script = f"""
import importlib
import json
from pathlib import Path
import sys
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))
{setup}
{imports}
from tools.evaluation.runtime_purity import audit_runtime_purity
print(json.dumps(audit_runtime_purity().as_dict()))
"""
    return _isolated(script)


def _clean_chain_args() -> dict[str, object]:
    return {
        "use_vision": True,
        "learned_reach": True,
        "reference_skills": (),
        "reach_reference_recovery_used": False,
        "reference_skill_calls": 0,
        "recovery_calls": 0,
        "vision": {
            "strict_mode": True,
            "v7_service_calls": 8,
            "invalid_frames": 0,
            "oracle_fallback_count": 0,
        },
        "pipeline_audit": {"all_prepared_skills_exercised": True},
        "runtime_purity": {
            "vendor_path_exposed": False,
            "loaded_v5_module_count": 0,
            "loaded_v5_modules": [],
            "import_blocker_enabled": True,
            "verified": True,
        },
    }


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def test_source_provenance_preserves_clean_pre_run_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "runtime-purity@example.invalid")
    _git(repository, "config", "user.name", "Runtime Purity Test")
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "--quiet", "--no-gpg-sign", "-m", "baseline")

    source_snapshot = capture_source_snapshot(repository)
    tracked.write_text("dirty after launch\n", encoding="utf-8")
    provenance = collect_evidence_provenance(
        SimpleNamespace(reach_checkpoint=None, checkpoints={}, artifact_lock=None),
        repository_root=repository,
        source_snapshot=source_snapshot,
    )

    assert provenance["source_commit"] == _git(repository, "rev-parse", "HEAD")
    assert provenance["source_worktree_clean_before_run"] is True
    assert provenance["source_clean_before_run"] is True
    assert provenance["source_git_status_before_run"] == []
    assert provenance["git_worktree_dirty"] is True
    assert provenance["git_status"] == [" M tracked.txt"]


def test_episode_runner_captures_source_before_isaac_launch() -> None:
    source = (ROOT / "tools" / "evaluation" / "episode_runner.py").read_text(
        encoding="utf-8"
    )
    capture = "source_snapshot = capture_source_snapshot(V7_ROOT)"
    launch = "app = app_launcher_class(args).app"
    assert source.index(capture) < source.index(launch)
    assert source.count("source_snapshot=source_snapshot") == 3


def test_runtime_purity_export_enables_blocker_and_records_source(tmp_path: Path) -> None:
    output = tmp_path / "runtime_purity.json"
    environment = os.environ.copy()
    environment.pop("STAGE_VLA_V5_ROOT", None)
    environment["PYTHONPATH"] = ""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.evaluation.export_runtime_purity",
            "--output",
            str(output),
            "--require-pure",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_commit"] == _git(ROOT, "rev-parse", "HEAD")
    assert payload["source_worktree_clean_before_run"] is (
        not payload["source_git_status_before_run"]
    )
    assert payload["source_clean_before_run"] is (
        payload["source_worktree_clean_before_run"]
    )
    assert payload["runtime_purity"]["import_blocker_enabled"] is True
    assert payload["runtime_purity"]["verified"] is True


def test_formal_runtime_has_no_v5_imports() -> None:
    violations: list[str] = []
    for root in FORMAL_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if _is_v5_name(name):
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {name}"
                        )
    assert violations == []


def test_formal_bootstrap_does_not_expose_v5() -> None:
    purity = _import_probe("tools.evaluation.bootstrap")
    assert purity["vendor_path_exposed"] is False
    assert purity["loaded_v5_module_count"] == 0


def test_importing_evaluator_does_not_load_v5() -> None:
    purity = _import_probe(
        "tools.evaluation.cli",
        "tools.evaluation.episode_runner",
    )
    assert purity["verified"] is True


def test_importing_stage_vla_v7_does_not_load_v5() -> None:
    purity = _import_probe("stage_vla_v7")
    assert purity["loaded_v5_modules"] == []


def test_action_evaluation_import_does_not_load_v5() -> None:
    purity = _import_probe("stage_vla_v7.action.evaluation")
    assert purity["loaded_v5_module_count"] == 0


def test_action_evaluation_public_api_has_no_v5_adapter() -> None:
    from stage_vla_v7.action import evaluation

    assert "legacy_vectorized_skill_success" not in evaluation.__all__
    assert not hasattr(evaluation, "legacy_vectorized_skill_success")


def test_runtime_purity_detects_loaded_v5_module() -> None:
    purity = audit_runtime_purity(
        modules={"stage_vla.rl.hidden": object()},
        search_path=(),
        environ={},
    )
    assert purity.loaded_v5_modules == ("stage_vla.rl.hidden",)
    assert purity.loaded_v5_module_count == 1
    assert purity.verified is False


def test_runtime_purity_detects_vendor_path() -> None:
    vendor = ROOT / "vendor" / "stage_vla_v5"
    purity = audit_runtime_purity(
        modules={},
        search_path=(str(vendor),),
        environ={},
    )
    assert purity.vendor_path_exposed is True
    assert purity.exposed_vendor_paths == (str(vendor.resolve()),)
    assert purity.verified is False


def test_runtime_purity_uses_configured_v5_root_for_path_detection(tmp_path: Path) -> None:
    configured = tmp_path / "external-v5"
    purity = audit_runtime_purity(
        modules={},
        search_path=(str(configured),),
        environ={"STAGE_VLA_V5_ROOT": str(configured)},
    )
    assert purity.vendor_path_exposed is True
    assert purity.v5_root_env_set is True


def test_runtime_purity_ignores_stage_vla_v7() -> None:
    purity = audit_runtime_purity(
        modules={
            "stage_vla_v7": object(),
            "stage_vla_v7.action": object(),
        },
        search_path=(str(ROOT / "src"),),
        environ={},
    )
    assert purity.loaded_v5_module_count == 0
    assert purity.vendor_path_exposed is False
    assert purity.verified is True


def test_require_v7_chain_rejects_loaded_v5() -> None:
    arguments = _clean_chain_args()
    arguments["runtime_purity"] = {
        "vendor_path_exposed": False,
        "loaded_v5_module_count": 1,
        "loaded_v5_modules": ["stage_vla"],
        "verified": False,
    }
    assert verify_v7_chain(**arguments) is False


def test_require_v7_chain_rejects_vendor_path() -> None:
    arguments = _clean_chain_args()
    arguments["runtime_purity"] = {
        "vendor_path_exposed": True,
        "loaded_v5_module_count": 0,
        "loaded_v5_modules": [],
        "verified": False,
    }
    assert verify_v7_chain(**arguments) is False


def test_require_v7_chain_accepts_clean_runtime() -> None:
    assert verify_v7_chain(**_clean_chain_args()) is True


def test_require_v7_chain_rejects_debug_oracle_even_without_fallback() -> None:
    arguments = _clean_chain_args()
    arguments["vision"] = {**arguments["vision"], "strict_mode": False}
    assert verify_v7_chain(**arguments) is False


def test_require_v7_chain_rejects_missing_import_blocker() -> None:
    arguments = _clean_chain_args()
    arguments["runtime_purity"] = {
        **arguments["runtime_purity"],
        "import_blocker_enabled": False,
    }
    assert verify_v7_chain(**arguments) is False


def test_runtime_import_blocker_rejects_v5_and_allows_v7() -> None:
    script = f"""
import importlib
import json
from pathlib import Path
import sys
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))
from tools.evaluation.runtime_purity import audit_runtime_purity, install_v5_import_blocker
install_v5_import_blocker()
importlib.import_module('stage_vla_v7.action.evaluation')
try:
    importlib.import_module('stage_vla.rl.v5_skill_contracts')
except ImportError as error:
    blocked = 'V5 Python imports are forbidden' in str(error)
else:
    blocked = False
print(json.dumps({{'blocked': blocked, **audit_runtime_purity().as_dict()}}))
"""
    result = _isolated(script)
    assert result["blocked"] is True
    assert result["import_blocker_enabled"] is True
    assert result["verified"] is True


def test_runtime_import_blocker_rejects_dirty_start(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "stage_vla.hidden_v5_probe", object())
    with pytest.raises(RuntimeError, match="already V5-exposed"):
        install_v5_import_blocker()


def test_vendor_absence_smoke_blocks_v5_and_imports_formal_stack() -> None:
    purity = _import_probe(
        "stage_vla_v7",
        "stage_vla_v7.action.evaluation",
        "tools.evaluation.cli",
        "tools.evaluation.episode_runner",
        block_v5=True,
    )
    assert purity["verified"] is True
