from __future__ import annotations

from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from stage_vla.action_output import ActionOutputModule as V5ActionOutputModule  # noqa: E402
from stage_vla.rl.v5_skill_contracts import reference_action as v5_reference_action  # noqa: E402
from stage_vla_v7.action import ActionOutputModule, reference_action  # noqa: E402
from stage_vla_v7.interfaces import SKILL_SEQUENCE, Skill  # noqa: E402


class _TensorActionSource:
    def __init__(self, action: torch.Tensor) -> None:
        self.action_tensor = action

    def action(self, _skill: object, observation: torch.Tensor) -> torch.Tensor:
        assert observation.shape[0] == self.action_tensor.shape[0]
        return self.action_tensor.clone()


def _reference_state(batch_size: int = 4) -> dict[str, object]:
    generator = torch.Generator().manual_seed(82017)
    support = torch.randn((batch_size, 3), generator=generator) * 0.02
    support[:, 2] = 0.0
    manipulated = support + torch.randn((batch_size, 3), generator=generator) * 0.03
    manipulated[:, 2] += 0.055
    ee = manipulated + torch.randn((batch_size, 3), generator=generator) * 0.015
    return {
        "ee": ee,
        "object": manipulated,
        "support": support,
        "left_tip": ee + torch.tensor([0.0, -0.03, -0.01]),
        "right_tip": ee + torch.tensor([0.0, 0.03, -0.01]),
        "grasp_target": manipulated + torch.tensor([0.002, -0.001, 0.0]),
        "speed": torch.rand(batch_size, generator=generator) * 0.08,
        "angular_speed": torch.rand(batch_size, generator=generator),
        "open": torch.tensor([True, False, True, False]),
        "held": torch.tensor([False, True, True, False]),
        "stack_height": torch.full((batch_size,), 0.0468),
    }


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_native_reference_action_matches_v5(skill: Skill) -> None:
    state = _reference_state()

    expected = v5_reference_action(
        skill.value,
        state,
        translation_limit_m=0.004,
        yaw_limit_rad=0.015,
    )
    actual = reference_action(
        skill,
        state,
        translation_limit_m=0.004,
        yaw_limit_rad=0.015,
    )

    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape == (4, 5)
    assert actual.dtype is expected.dtype is torch.float32
    assert bool(torch.isfinite(actual).all())
    assert float((actual - expected).abs().max()) == 0.0


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_native_policy_output_matches_v5_projection_and_finished_mask(skill: Skill) -> None:
    observation = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    candidate = torch.tensor(
        [
            [1.5, -1.5, 0.25, 2.0, 0.75],
            [-0.1, 0.2, -0.3, 0.4, -0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [float("-inf"), 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidate[3, 0] = -2.0
    finished = torch.tensor([False, True, False, True])
    source = _TensorActionSource(candidate)
    legacy = V5ActionOutputModule(source)
    native = ActionOutputModule(source)

    expected = legacy.emit(skill.value, observation, finished=finished)
    actual = native.emit(skill, observation, finished=finished)

    assert actual.skill.value == expected.skill.value == skill.value
    assert torch.equal(actual.command, expected.command)
    assert torch.equal(actual.candidate, expected.candidate)
    assert torch.equal(actual.active, expected.active)
    assert actual.source == expected.source == "policy"
    assert actual.reference is expected.reference is None
    assert actual.action_order == expected.action_order == ("dx", "dy", "dz", "dyaw", "grip")


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_native_reference_output_matches_v5(skill: Skill) -> None:
    observation = torch.zeros((4, 7), dtype=torch.float32)
    state = _reference_state()
    finished = torch.tensor([False, False, True, True])
    legacy = V5ActionOutputModule(None, reference_skills=(skill.value,))
    native = ActionOutputModule(None, reference_skills=(skill,))

    expected = legacy.emit(
        skill.value,
        observation,
        reference_state=state,
        finished=finished,
        translation_limit_m=0.003,
        yaw_limit_rad=0.01,
    )
    actual = native.emit(
        skill,
        observation,
        reference_state=state,
        finished=finished,
        translation_limit_m=0.003,
        yaw_limit_rad=0.01,
    )

    assert torch.equal(actual.command, expected.command)
    assert torch.equal(actual.candidate, expected.candidate)
    assert torch.equal(actual.active, expected.active)
    assert actual.source == expected.source == "reference"
    assert actual.reference is not None and expected.reference is not None
    assert torch.equal(actual.reference, expected.reference)


def test_native_policy_output_can_emit_matching_dagger_reference() -> None:
    observation = torch.zeros((4, 7), dtype=torch.float32)
    candidate = torch.linspace(-1.5, 1.5, 20).reshape(4, 5)
    state = _reference_state()
    source = _TensorActionSource(candidate)

    expected = V5ActionOutputModule(source).emit(
        "ALIGN",
        observation,
        reference_state=state,
        include_reference=True,
    )
    actual = ActionOutputModule(source).emit(
        Skill.ALIGN,
        observation,
        reference_state=state,
        include_reference=True,
    )

    assert torch.equal(actual.command, expected.command)
    assert actual.reference is not None and expected.reference is not None
    assert torch.equal(actual.reference, expected.reference)


def test_physical_runtime_uses_native_action_output_module() -> None:
    source = (ROOT / "tools" / "evaluation" / "episode_runner.py").read_text(
        encoding="utf-8"
    )

    assert "stage_vla.action_output" not in source
    assert "from stage_vla_v7.action import ActionOutputModule" in source
