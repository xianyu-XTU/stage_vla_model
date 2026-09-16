from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from stage_vla_v7.interfaces import PhysicalState
from stage_vla_v7.simulation.environments import (
    KnownSizeObservationInput,
    build_known_size_observation,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.known_size_grasp_vecenv import (  # noqa: E402
    KnownSizeGraspVecEnv as V5KnownSizeGraspVecEnv,
)


def _sample(seed: int, count: int = 5):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    measured = {
        "red": rand(count, 3),
        "blue": rand(count, 3),
        "vel": rand(count, 3),
        "angular": rand(count, 3),
        "red_quat": rand(count, 4),
        "blue_quat": rand(count, 4),
        "open": torch.rand(count, generator=generator) > 0.5,
        "grip": torch.rand(count, 2, generator=generator) * 0.04,
        "speed": torch.rand(count, generator=generator),
        "ee": rand(count, 3),
        "left_tip": rand(count, 3),
        "right_tip": rand(count, 3),
        "force": torch.rand(count, 2, generator=generator) * 10.0,
        "physical": torch.rand(count, generator=generator) > 0.5,
        "between_fingertips": torch.rand(count, generator=generator) > 0.5,
        "left_height_aligned": torch.rand(count, generator=generator) > 0.5,
        "right_height_aligned": torch.rand(count, generator=generator) > 0.5,
        "finger_a_contact": torch.rand(count, generator=generator) > 0.5,
        "finger_b_contact": torch.rand(count, generator=generator) > 0.5,
        "grasp_target": rand(count, 3),
        "radial_error_m": torch.rand(count, generator=generator),
        "left_height_error_m": torch.rand(count, generator=generator),
        "right_height_error_m": torch.rand(count, generator=generator),
        "projection_alpha": torch.rand(count, generator=generator),
        "fingertip_gap_m": torch.rand(count, generator=generator),
        "q": rand(count, 7),
        "qd": rand(count, 7),
    }
    values = {
        "target_force": torch.rand(count, generator=generator) * 8.0 + 1.0,
        "previous_force": torch.rand(count, 2, generator=generator) * 10.0,
        "previous_action": rand(count, 5).clamp(-1.0, 1.0),
        "stable_count": torch.randint(0, 4, (count,), generator=generator),
        "steps": torch.randint(0, 100, (count,), generator=generator),
        "size": torch.rand(count, 3, generator=generator),
        "context": torch.rand(count, 10, generator=generator),
        "visual_red": rand(count, 3),
        "visual_blue": rand(count, 3),
    }
    return measured, values


def _v5_observation(
    measured: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    *,
    visual: bool,
    context: bool,
) -> torch.Tensor:
    count = len(measured["red"])
    physical_batch = SimpleNamespace(action_context=lambda: values["context"])
    fake = SimpleNamespace(
        measured=measured,
        gripper_action=SimpleNamespace(target_force_n=values["target_force"]),
        prev_force=values["previous_force"],
        visual_red=values["visual_red"] if visual else None,
        visual_blue=values["visual_blue"] if visual else None,
        unwrapped=SimpleNamespace(step_dt=0.02),
        geometry_adapter=SimpleNamespace(legacy_size_features=lambda: values["size"]),
        prev_unit=values["previous_action"],
        stable_count=values["stable_count"],
        stable_steps=3,
        steps=values["steps"],
        episode_steps=100,
        known_size=SimpleNamespace(max_force_n=25.0),
        include_physical_context=context,
        physical_batch=physical_batch,
        num_envs=count,
        observation_dim=65 if context else 55,
    )
    return V5KnownSizeGraspVecEnv._obs(fake)["policy"]


@pytest.mark.parametrize("seed", [0, 7, 61081])
@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("context", [False, True])
def test_known_size_observation_matches_v5(
    seed: int,
    visual: bool,
    context: bool,
) -> None:
    measured, values = _sample(seed)
    state = PhysicalState.from_legacy_mapping(measured)
    actual = build_known_size_observation(KnownSizeObservationInput(
        state=state,
        target_force_n=values["target_force"],
        max_force_n=25.0,
        size_features=values["size"],
        previous_action=values["previous_action"],
        previous_force=values["previous_force"],
        stable_count=values["stable_count"],
        stable_steps=3,
        step_count=values["steps"],
        episode_steps=100,
        step_dt_s=0.02,
        visual_object_position=values["visual_red"] if visual else None,
        visual_support_position=values["visual_blue"] if visual else None,
        physical_context=values["context"] if context else None,
    ))
    expected = _v5_observation(measured, values, visual=visual, context=context)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
    assert float((actual - expected).abs().max()) == 0.0


def test_physical_state_legacy_mapping_round_trip() -> None:
    measured, _ = _sample(11, count=2)
    state = PhysicalState.from_legacy_mapping(measured)
    round_trip = state.to_legacy_mapping()

    assert set(round_trip) == set(measured)
    assert all(round_trip[key] is value for key, value in measured.items())


def test_known_size_observation_rejects_half_visual_override() -> None:
    measured, values = _sample(3, count=1)
    with pytest.raises(ValueError, match="must be supplied together"):
        build_known_size_observation(KnownSizeObservationInput(
            state=PhysicalState.from_legacy_mapping(measured),
            target_force_n=values["target_force"],
            max_force_n=25.0,
            size_features=values["size"],
            previous_action=values["previous_action"],
            previous_force=values["previous_force"],
            stable_count=values["stable_count"],
            stable_steps=3,
            step_count=values["steps"],
            episode_steps=100,
            step_dt_s=0.02,
            visual_object_position=values["visual_red"],
        ))
