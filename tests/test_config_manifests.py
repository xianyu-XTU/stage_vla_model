from __future__ import annotations

import json
from pathlib import Path

from stage_vla_v7.interfaces import SKILL_SEQUENCE


ROOT = Path(__file__).parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_split_component_configs_exist_and_are_valid_json() -> None:
    components = _json(ROOT / "config" / "components.example.json")
    canonical = components["canonical_configs"]
    assert isinstance(canonical, dict)
    assert set(canonical) == {"vision", "language", "action", "simulation", "task"}
    for relative in canonical.values():
        manifest = _json(ROOT / "config" / str(relative))
        assert str(manifest["schema"]).startswith("stage_vla_v7.")


def test_artifact_lock_describes_every_skill_checkpoint() -> None:
    lock = _json(ROOT / "config" / "artifacts.lock.json")
    artifacts = lock["artifacts"]
    assert isinstance(artifacts, dict)
    for skill in SKILL_SEQUENCE:
        record = artifacts[skill.value.lower()]
        assert record["skill"] == skill.value
        assert record["model_type"] == "torchscript_parameter_policy"
        assert record["action_dim"] == 5
        assert record["observation_dim"] == (52 if skill.value == "REACH" else 55)
        assert len(record["sha256"]) == 64
        assert record["logical_path"].endswith("/policy.ts")
