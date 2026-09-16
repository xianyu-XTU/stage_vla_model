from __future__ import annotations

from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.v5_skill_contracts import (  # noqa: E402
    skill_failure as v5_skill_failure,
    skill_success as v5_skill_success,
)
from stage_vla_v7.action.evaluation.vectorized_success import (  # noqa: E402
    vectorized_skill_failure,
    vectorized_skill_success,
)
from stage_vla_v7.interfaces import SKILL_SEQUENCE, Skill  # noqa: E402


def _scenario_batch(skill: Skill, *, aliases: bool) -> tuple[dict[str, object], torch.Tensor]:
    """Return success, failure, in-progress, and exact-boundary rows."""
    support = torch.zeros((4, 3), dtype=torch.float64)
    manipulated = torch.tensor(
        [[0.0, 0.0, 0.0468]] * 4,
        dtype=torch.float64,
    )
    ee = manipulated + torch.tensor([0.0, 0.0, 0.010], dtype=torch.float64)
    left_tip = manipulated + torch.tensor([0.0, 0.0, 0.010], dtype=torch.float64)
    right_tip = left_tip.clone()
    grasp_target = manipulated.clone()
    speed = torch.zeros(4, dtype=torch.float64)
    angular_speed = torch.zeros(4, dtype=torch.float64)
    opened = torch.ones(4, dtype=torch.bool)
    held = torch.ones(4, dtype=torch.bool)
    stable_count = torch.tensor([3, 3, 1, 3], dtype=torch.int32)

    # Row 1 is a terminal failure. Row 2 remains valid but incomplete.
    if skill is Skill.REACH:
        manipulated[1] = torch.tensor([0.50, 0.0, 0.0468])
        ee[1] = torch.tensor([0.0, 0.0, 0.0468])
        left_tip[1] = ee[1]
        right_tip[1] = ee[1]
        grasp_target[1] = manipulated[1]
        left_tip[2, 0] += 0.020
        right_tip[2, 0] += 0.020
        left_tip[3, 0] += 0.012
        right_tip[3, 0] += 0.012
    elif skill is Skill.LIFT:
        manipulated[0, 2] = 0.070
        ee[0] = manipulated[0] + torch.tensor([0.0, 0.0, 0.010])
        held[1] = False
        manipulated[3, 2] = 0.060
    elif skill is Skill.TRANSPORT:
        manipulated[0] = torch.tensor([0.020, 0.0, 0.080])
        held[1] = False
        manipulated[2, 0] = 0.080
        manipulated[3, 0] = 0.045
    elif skill is Skill.ALIGN:
        manipulated[0] = torch.tensor([0.005, 0.0, 0.0518])
        held[1] = False
        manipulated[2, 0] = 0.020
        manipulated[3, 0] = 0.010
    elif skill is Skill.DESCEND:
        manipulated[0] = torch.tensor([0.005, 0.0, 0.0488])
        held[1] = False
        manipulated[2, 2] = 0.0568
        manipulated[3, 2] = 0.0508
    elif skill is Skill.RELEASE_STABILIZE:
        manipulated[0] = torch.tensor([0.020, 0.0, 0.0518])
        manipulated[1, 2] = 0.010
        opened[2] = False
        manipulated[3, 2] = 0.0568
    elif skill is Skill.RETREAT:
        ee[0] = manipulated[0] + torch.tensor([0.0, 0.0, 0.110])
        manipulated[1, 2] = 0.010
        ee[1] = manipulated[1]
        ee[2] = manipulated[2] + torch.tensor([0.0, 0.0, 0.050])
        ee[3] = manipulated[3] + torch.tensor([0.0, 0.0, 0.100])
    else:
        held[1] = False
        ee[3] = manipulated[3] + torch.tensor([0.022, 0.0, 0.0])

    object_key, support_key = (("red", "blue") if aliases else ("object", "support"))
    state: dict[str, object] = {
        "ee": ee,
        object_key: manipulated,
        support_key: support,
        "left_tip": left_tip,
        "right_tip": right_tip,
        "grasp_target": grasp_target,
        "speed": speed,
        "angular_speed": angular_speed,
        "open": opened,
        "held": held,
        "stack_height": torch.full((4,), 0.0468, dtype=torch.float64),
    }
    return state, stable_count


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
@pytest.mark.parametrize("aliases", [False, True], ids=["roles", "legacy_aliases"])
def test_native_success_and_failure_match_v5_fixed_scenarios(
    skill: Skill,
    aliases: bool,
) -> None:
    state, stable_count = _scenario_batch(skill, aliases=aliases)

    expected_success = v5_skill_success(
        skill.value,
        state,
        stable_count=stable_count,
        stable_steps=3,
    )
    actual_success = vectorized_skill_success(
        skill,
        state,
        stable_count=stable_count,
        stable_steps=3,
    )
    expected_failure = v5_skill_failure(skill.value, state)
    actual_failure = vectorized_skill_failure(skill, state)

    assert torch.equal(actual_success, expected_success)
    assert torch.equal(actual_failure, expected_failure)
    assert actual_success.shape == expected_success.shape == (4,)
    assert actual_failure.shape == expected_failure.shape == (4,)
    assert actual_success.dtype is expected_success.dtype is torch.bool
    assert actual_failure.dtype is expected_failure.dtype is torch.bool
    assert bool(torch.isfinite(actual_success.float()).all())
    assert bool(torch.isfinite(actual_failure.float()).all())
    assert bool(actual_success[0])
    assert bool(actual_failure[1])
    assert not bool(actual_success[2])
    assert not bool(actual_failure[2])


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_native_success_matches_v5_randomized_batches(skill: Skill) -> None:
    generator = torch.Generator().manual_seed(7300 + list(SKILL_SEQUENCE).index(skill))
    batch_size = 257
    state: dict[str, object] = {
        "ee": torch.randn((batch_size, 3), generator=generator, dtype=torch.float64) * 0.15,
        "red": torch.randn((batch_size, 3), generator=generator, dtype=torch.float64) * 0.15,
        "blue": torch.randn((batch_size, 3), generator=generator, dtype=torch.float64) * 0.15,
        "left_tip": torch.randn((batch_size, 3), generator=generator, dtype=torch.float64) * 0.15,
        "right_tip": torch.randn((batch_size, 3), generator=generator, dtype=torch.float64) * 0.15,
        "grasp_target": torch.randn((batch_size, 3), generator=generator, dtype=torch.float64) * 0.15,
        "speed": torch.rand(batch_size, generator=generator, dtype=torch.float64) * 0.10,
        "angular_speed": torch.rand(batch_size, generator=generator, dtype=torch.float64) * 2.0,
        "open": torch.rand(batch_size, generator=generator) > 0.5,
        "held": torch.rand(batch_size, generator=generator) > 0.5,
        "stack_height": torch.full((batch_size,), 0.0468, dtype=torch.float64),
    }
    stable_count = torch.randint(0, 6, (batch_size,), generator=generator)

    expected_success = v5_skill_success(
        skill.value,
        state,
        stable_count=stable_count,
        stable_steps=3,
    )
    actual_success = vectorized_skill_success(
        skill,
        state,
        stable_count=stable_count,
        stable_steps=3,
    )
    expected_failure = v5_skill_failure(skill.value, state)
    actual_failure = vectorized_skill_failure(skill, state)

    assert torch.equal(actual_success, expected_success)
    assert torch.equal(actual_failure, expected_failure)


def test_physical_runtime_uses_native_success_checker() -> None:
    source = (ROOT / "tools" / "evaluation" / "episode_runner.py").read_text(
        encoding="utf-8"
    )

    assert "legacy_vectorized_skill_success" not in source
    assert "SuccessChecker" in source
