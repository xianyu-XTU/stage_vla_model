from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from stage_vla_v7.action import ACTION_REGISTRY, TorchScriptActionPolicy
from stage_vla_v7.action.network import load_torchscript_checkpoint
from stage_vla_v7.interfaces import ActionRequest, SKILL_SEQUENCE, Skill
from stage_vla_v7.orchestration import rigid_cube_profile


class _FiveParameterNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("output_values", torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5]))

    def forward(self, observation):
        return self.output_values.expand(observation.shape[0], -1)


@pytest.mark.parametrize("skill", SKILL_SEQUENCE)
def test_each_skill_loads_a_five_parameter_torchscript_policy(tmp_path, skill: Skill) -> None:
    checkpoint = tmp_path / f"{skill.value.lower()}.ts"
    definition = ACTION_REGISTRY.require(skill)
    torch.jit.trace(
        _FiveParameterNetwork(),
        torch.zeros((1, definition.legacy_observation_dim)),
    ).save(str(checkpoint))
    policy = TorchScriptActionPolicy(
        checkpoint,
        skill=skill,
        observation_dim=definition.legacy_observation_dim,
    )
    result = policy.predict(
        ActionRequest(
            skill,
            (0.0,) * definition.legacy_observation_dim,
            rigid_cube_profile("red_cube"),
            rigid_cube_profile("blue_cube"),
        )
    )
    assert result.action.values == pytest.approx((0.1, -0.2, 0.3, -0.4, 0.5))


def test_checkpoint_hash_mismatch_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "policy.ts"
    torch.jit.trace(_FiveParameterNetwork(), torch.zeros((1, 3))).save(str(checkpoint))
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert load_torchscript_checkpoint(checkpoint, expected_sha256=actual).path == checkpoint
    with pytest.raises(ValueError, match="SHA256"):
        load_torchscript_checkpoint(checkpoint, expected_sha256="0" * 64)
