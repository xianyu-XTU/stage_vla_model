from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "stage_vla_v7"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_parent = path.relative_to(SOURCE).parent.parts
    current_package = ("stage_vla_v7", *relative_parent)
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(current_package) - (node.level - 1)
                prefix = current_package[:keep]
                suffix = tuple((node.module or "").split(".")) if node.module else ()
                result.append(".".join((*prefix, *suffix)))
            else:
                result.append(node.module or "")
    return tuple(result)


def _package_imports(package: str) -> list[tuple[Path, str]]:
    return [
        (path, imported)
        for path in (SOURCE / package).rglob("*.py")
        for imported in _imports(path)
    ]


def _assert_no_prefix(package: str, prefixes: tuple[str, ...]) -> None:
    violations = [
        f"{path.relative_to(SOURCE)} -> {imported}"
        for path, imported in _package_imports(package)
        if imported.startswith(prefixes)
    ]
    assert violations == []


def test_interfaces_are_dependency_free() -> None:
    _assert_no_prefix(
        "interfaces",
        (
            "torch",
            "numpy",
            "cv2",
            "transformers",
            "isaaclab",
            "isaacsim",
            "omni",
            "stage_vla_v7.action",
            "stage_vla_v7.language",
            "stage_vla_v7.vision",
            "stage_vla_v7.orchestration",
            "stage_vla_v7.simulation",
        ),
    )


def test_vision_language_and_action_are_mutually_isolated() -> None:
    _assert_no_prefix(
        "vision",
        ("stage_vla_v7.language", "stage_vla_v7.action", "stage_vla_v7.simulation"),
    )
    _assert_no_prefix(
        "language",
        ("stage_vla_v7.vision", "stage_vla_v7.action", "stage_vla_v7.simulation"),
    )
    _assert_no_prefix(
        "action",
        ("stage_vla_v7.vision", "stage_vla_v7.language", "stage_vla_v7.simulation"),
    )


def test_action_and_orchestration_do_not_import_simulator_implementations() -> None:
    simulator_prefixes = (
        "isaaclab",
        "isaacsim",
        "omni",
        "gymnasium",
        "stage_vla_v7.simulation",
    )
    _assert_no_prefix("action", simulator_prefixes)
    _assert_no_prefix("orchestration", simulator_prefixes)


def test_pipeline_has_no_simulation_dependency() -> None:
    pipeline = SOURCE / "orchestration" / "pipeline.py"
    assert not any(
        imported.startswith(("stage_vla_v7.simulation", "isaaclab", "isaacsim", "omni"))
        for imported in _imports(pipeline)
    )
