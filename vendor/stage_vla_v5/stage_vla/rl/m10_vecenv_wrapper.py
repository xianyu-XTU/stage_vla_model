"""RSL-RL wrapper for M10 Stage-aware Action-DSL PPO.

The M9-B DSL decoder/edge-yaw/action stack is inherited unchanged. M10 adds a
5-D one-hot stage condition. M10-12-F1 additionally exports the *pre-action*
base release-ready mask through ``extras`` so the custom PPO can localize the
direct terminal policy gradient to GRIP without changing environment actions.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .m10_factor_local_credit import (
    M10_F1_CREDIT_SEMANTICS,
    M10_F1_READY_EXTRAS_KEY,
)
from .m10_stage_core import M10_STAGE_COUNT, append_stage_one_hot
from .m9b_vecenv_wrapper import M9BActionDSLVecEnvWrapper


class M10StageAwareActionDSLVecEnvWrapper(M9BActionDSLVecEnvWrapper):
    """M9-B actions + current five-stage one-hot observation + F1 ready metadata."""

    def __init__(self, env, clip_actions: float | None = None, *, config_file: Path) -> None:
        super().__init__(env, clip_actions=clip_actions, config_file=config_file)
        term_cfg = self.unwrapped.reward_manager.get_term_cfg("stage_aware")
        term = term_cfg.func
        if not hasattr(term, "stage"):
            raise RuntimeError("M10 stage_aware RewardManager term does not expose current stage state")
        self.stage_reward_term = term
        self.stage_observation_dim = M10_STAGE_COUNT
        self.base_policy_observation_dim: int | None = None
        self.policy_observation_dim: int | None = None
        self.last_stage_one_hot: torch.Tensor | None = None
        self.factor_local_credit_semantics = M10_F1_CREDIT_SEMANTICS
        self.last_release_ready_before_action = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _augment(self, observations):
        if "policy" not in observations:
            raise KeyError("M10 requires a 'policy' observation group")
        policy = observations["policy"]
        if not torch.is_tensor(policy):
            policy = torch.as_tensor(policy, device=self.device)
        stage = self.stage_reward_term.stage.to(device=policy.device)
        augmented = append_stage_one_hot(policy, stage)
        observations["policy"] = augmented
        self.base_policy_observation_dim = int(policy.shape[-1])
        self.policy_observation_dim = int(augmented.shape[-1])
        self.last_stage_one_hot = augmented[..., -M10_STAGE_COUNT:].detach().clone()
        return observations

    def get_observations(self):
        return self._augment(super().get_observations())

    def _before_env_step(self, decoded) -> None:
        # Snapshot the exact state used by T1's direct OPEN command credit before
        # env.step() updates the stateful reward term to s_(t+1).
        ready_before_action = self.stage_reward_term.last_release_ready.clone()
        self.last_release_ready_before_action.copy_(ready_before_action)
        self.stage_reward_term.set_policy_gripper_token(
            decoded.token_indices[..., 3],
            release_ready_before_action=ready_before_action,
        )

    def step(self, actions):
        # M9-B base wrapper executes the raw 7-D pose-IK action unchanged. H4X's
        # terminal XYZ execution hold is intentionally inactive in F1.
        observations, reward, dones, extras = super().step(actions)
        if not isinstance(extras, dict):
            raise TypeError("M10-12-F1 requires dict extras from the RSL-RL env wrapper")
        extras[M10_F1_READY_EXTRAS_KEY] = self.last_release_ready_before_action.clone()
        return self._augment(observations), reward, dones, extras
