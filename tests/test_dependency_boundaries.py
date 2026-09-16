from __future__ import annotations

import ast
from pathlib import Path


def test_model_packages_do_not_import_each_other() -> None:
    source = Path(__file__).parents[1] / "src" / "stage_vla_v7"
    forbidden = {
        "vision": ("stage_vla_v7.language", "stage_vla_v7.action"),
        "language": ("stage_vla_v7.vision", "stage_vla_v7.action"),
        "action": ("stage_vla_v7.vision", "stage_vla_v7.language"),
    }
    violations = []
    for package, prefixes in forbidden.items():
        for path in (source / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name.startswith(prefixes) for name in names):
                    violations.append(f"{path.name}: {names}")
    assert violations == []
