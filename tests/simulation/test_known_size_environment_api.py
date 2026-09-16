from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stage_vla_v7.simulation.isaac_lab.known_size_environment import (
    KnownSizeGraspEnvironment,
)
from stage_vla_v7.simulation.physics import PhysicalGraspConfig


class _EnvironmentHarness:
    def __init__(self, sampled_states: list[dict[str, torch.Tensor]]) -> None:
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.skill = "GRASP"
        self.arm_locked = True
        self.episode_steps = 80
        self.max_episode_length = 80
        self.stable_steps = 3
        self.steps = torch.tensor([4, 5])
        self.stable_count = torch.tensor([2, 1])
        self.finished = torch.tensor([True, True])
        self.last_success = torch.tensor([True, False])
        self.last_failure = torch.tensor([False, True])
        self.last_timeout = torch.tensor([False, False])
        self.prev_unit = torch.ones(2, 5)
        self.prev_force = torch.zeros(2, 2)
        self.entry_red_z = torch.zeros(2)
        self.lift_target_height_m = torch.full((2,), 0.06)
        self.snapshot_recovery_active = True
        self.measured = sampled_states[0]
        self._sampled_states = iter(sampled_states)
        self.lift_target_updates = 0

    def get_physical_state(self) -> dict[str, torch.Tensor]:
        return next(self._sampled_states)

    def mark_continuous_handoff(self) -> None:
        KnownSizeGraspEnvironment.mark_continuous_handoff(self)

    def update_lift_target_from_support(self) -> torch.Tensor:
        self.lift_target_updates += 1
        return self.lift_target_height_m.clone()

    def observe(self) -> str:
        return "policy-observation"


def _sampled_state(*, object_z: tuple[float, float] = (0.10, 0.11)):
    return {
        "red": torch.tensor([[0.0, 0.0, object_z[0]], [0.0, 0.0, object_z[1]]]),
        "force": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    }


def test_configure_skill_preserves_the_inline_handoff_state_transition() -> None:
    before = _sampled_state()
    after = {name: value.clone() for name, value in before.items()}
    env = _EnvironmentHarness([before, after])

    observation, exact = KnownSizeGraspEnvironment.configure_skill(
        env,
        "LIFT",
        episode_steps=120,
        stable_steps=4,
        active_mask=torch.tensor([True, False]),
    )

    assert observation == "policy-observation"
    assert exact is True
    assert env.skill == "LIFT"
    assert env.arm_locked is False
    assert env.episode_steps == env.max_episode_length == 120
    assert env.stable_steps == 4
    assert env.steps.tolist() == [0, 0]
    assert env.stable_count.tolist() == [0, 0]
    assert env.finished.tolist() == [False, True]
    assert not env.last_success.any()
    assert not env.last_failure.any()
    assert not env.last_timeout.any()
    assert not env.prev_unit.any()
    assert torch.equal(env.prev_force, before["force"])
    assert env.snapshot_recovery_active is False
    assert env.measured is after
    assert env.lift_target_updates == 1


def test_configure_skill_rejects_a_physical_handoff_change() -> None:
    before = _sampled_state()
    after = _sampled_state(object_z=(0.10, 0.12))
    env = _EnvironmentHarness([before, after])

    with pytest.raises(RuntimeError, match="physical state changed during handoff"):
        KnownSizeGraspEnvironment.configure_skill(
            env,
            "ALIGN",
            episode_steps=100,
            stable_steps=3,
            active_mask=torch.ones(2, dtype=torch.bool),
        )


def test_external_step_synchronization_matches_reach_exit_bookkeeping() -> None:
    measured = _sampled_state(object_z=(0.14, 0.17))
    env = _EnvironmentHarness([measured])
    env.measured = _sampled_state()

    result = KnownSizeGraspEnvironment.synchronize_after_external_step(env)

    assert result is measured
    assert env.measured is measured
    assert torch.equal(env.entry_red_z, measured["red"][:, 2])
    assert torch.equal(env.prev_force, measured["force"])


def test_evaluation_configuration_preserves_the_existing_tolerance_update() -> None:
    env = SimpleNamespace(
        auto_reset=True,
        physical_cfg=PhysicalGraspConfig(
            radial_tolerance_m=0.03,
            height_tolerance_m=0.012,
            contact_force_threshold_n=0.5,
            endpoint_margin=0.05,
        ),
    )

    KnownSizeGraspEnvironment.configure_evaluation(
        env,
        auto_reset=False,
        physical_height_tolerance_m=0.015,
    )

    assert env.auto_reset is False
    assert env.physical_cfg.height_tolerance_m == pytest.approx(0.015)
    assert env.physical_cfg.radial_tolerance_m == pytest.approx(0.03)
