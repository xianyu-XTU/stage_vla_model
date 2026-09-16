"""Demonstration-guided pretraining for independent StagePPO skills.

This module deliberately keeps HER data outside PPO's on-policy rollout.  A
skill actor is first trained off-policy with demonstrations, future-goal HER,
and a Q-filtered behaviour-cloning loss.  Only the actor MLP is then copied to
the matching StagePPO actor; PPO starts with a fresh critic, optimizer, and
rollout storage.

The first supported contract is ALIGN (52 observations, 5 actions).  Its first
three observation values are the metric goal error divided by 0.05 m, so HER
can relabel those values without changing the deployed StagePPO interface.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class StagePPOFDConfig:
    observation_dim: int = 52
    action_dim: int = 5
    goal_dim: int = 3
    goal_observation_start: int = 0
    goal_scale_m: float = 0.05
    xy_success_m: float = 0.010
    z_success_m: float = 0.0115
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    bc_weight: float = 1.0
    action_l2_weight: float = 1e-4

    def validate(self) -> None:
        integers = (self.observation_dim, self.action_dim, self.goal_dim,
                    self.goal_observation_start)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in integers):
            raise ValueError("dimensions and goal offset must be integers")
        if min(self.observation_dim, self.action_dim, self.goal_dim) < 1:
            raise ValueError("dimensions must be positive")
        if self.goal_dim != 3:
            raise ValueError("the initial ALIGN adapter requires an XYZ goal")
        if self.goal_observation_start < 0 or (
                self.goal_observation_start + self.goal_dim > self.observation_dim):
            raise ValueError("goal slice lies outside the observation")
        positive = (self.goal_scale_m, self.xy_success_m, self.z_success_m,
                    self.tau, self.actor_lr, self.critic_lr)
        if any(not torch.isfinite(torch.tensor(v)) or v <= 0 for v in positive):
            raise ValueError("scales, tolerances, tau, and learning rates must be positive")
        if not 0 < self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau are outside their valid ranges")
        if min(self.bc_weight, self.action_l2_weight) < 0:
            raise ValueError("loss weights cannot be negative")


@dataclass(frozen=True)
class TransitionBatch:
    """One stage's transitions, including enough truth to relabel safely."""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    achieved_goal: torch.Tensor
    next_achieved_goal: torch.Tensor
    desired_goal: torch.Tensor
    physical_failure: torch.Tensor
    timeout: torch.Tensor
    task_valid: torch.Tensor

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def validate(self, cfg: StagePPOFDConfig) -> None:
        cfg.validate()
        n = len(self)
        expected = {
            "obs": (n, cfg.observation_dim),
            "actions": (n, cfg.action_dim),
            "rewards": (n,),
            "next_obs": (n, cfg.observation_dim),
            "achieved_goal": (n, cfg.goal_dim),
            "next_achieved_goal": (n, cfg.goal_dim),
            "desired_goal": (n, cfg.goal_dim),
            "physical_failure": (n,),
            "timeout": (n,),
            "task_valid": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype == torch.bool:
                continue
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in ("physical_failure", "timeout", "task_valid"):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        if (self.actions.abs() > 1 + 1e-6).any():
            raise ValueError("demonstration actions must be normalized to [-1, 1]")
        if (self.physical_failure & self.timeout).any():
            raise ValueError("physical failure and timeout must be exclusive")

    def index(self, ids: torch.Tensor) -> "TransitionBatch":
        return replace(self, **{name: getattr(self, name)[ids] for name in self.__dataclass_fields__})

    @staticmethod
    def concatenate(parts: Iterable["TransitionBatch"]) -> "TransitionBatch":
        items = list(parts)
        if not items:
            raise ValueError("at least one transition batch is required")
        return TransitionBatch(**{
            name: torch.cat([getattr(item, name) for item in items], dim=0)
            for name in items[0].__dataclass_fields__
        })


def _goal_error(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    if achieved.shape != desired.shape or achieved.ndim != 2 or achieved.shape[1] != 3:
        raise ValueError("achieved and desired goals must both be [N,3]")
    return achieved - desired


def goal_success(next_achieved: torch.Tensor, desired: torch.Tensor,
                 task_valid: torch.Tensor, cfg: StagePPOFDConfig) -> torch.Tensor:
    """ALIGN geometric success plus non-relabeled grasp/stability validity."""
    error = _goal_error(next_achieved, desired)
    return ((error[:, :2].norm(dim=-1) < cfg.xy_success_m)
            & (error[:, 2].abs() < cfg.z_success_m) & task_valid)


def write_goal_error(obs: torch.Tensor, achieved: torch.Tensor,
                     desired: torch.Tensor, cfg: StagePPOFDConfig) -> torch.Tensor:
    """Rewrite only the declared goal-error observation slice."""
    if obs.ndim != 2 or obs.shape[1] != cfg.observation_dim:
        raise ValueError("observation does not match the StagePPO skill contract")
    result = obs.clone()
    start, end = cfg.goal_observation_start, cfg.goal_observation_start + cfg.goal_dim
    result[:, start:end] = _goal_error(achieved, desired) / cfg.goal_scale_m
    return result


def relabel_future(batch: TransitionBatch, future_indices: torch.Tensor,
                   cfg: StagePPOFDConfig) -> TransitionBatch:
    """Apply future-goal HER to one time-ordered trajectory.

    ``future_indices[i]`` points to a transition at or after ``i`` in the same
    episode.  Grasp/stability validity and real physical failures are retained;
    HER is not allowed to turn an invalid grasp or a fallen block into success.
    """
    batch.validate(cfg)
    n = len(batch)
    if (not isinstance(future_indices, torch.Tensor) or future_indices.dtype != torch.long
            or tuple(future_indices.shape) != (n,)):
        raise ValueError("future_indices must be an int64 [N] tensor")
    steps = torch.arange(n, device=future_indices.device)
    if (future_indices < steps).any() or (future_indices >= n).any():
        raise ValueError("HER goals must come from the current or a future transition")
    goal = batch.next_achieved_goal[future_indices]
    obs = write_goal_error(batch.obs, batch.achieved_goal, goal, cfg)
    next_obs = write_goal_error(batch.next_obs, batch.next_achieved_goal, goal, cfg)
    success = (goal_success(batch.next_achieved_goal, goal, batch.task_valid, cfg)
               & ~batch.physical_failure & ~batch.timeout)
    # Sparse reward follows the paper's convention. Timeouts remain terminal;
    # physical failures are always terminal and cannot be relabeled away.
    rewards = torch.where(success, torch.zeros_like(batch.rewards), -torch.ones_like(batch.rewards))
    dones = success | batch.physical_failure | batch.timeout
    # TransitionBatch stores terminal causes instead of an ambiguous done bit.
    # A relabeled success is represented by reward==0; real causes stay intact.
    result = replace(batch, obs=obs, next_obs=next_obs, rewards=rewards,
                     desired_goal=goal, physical_failure=batch.physical_failure,
                     timeout=batch.timeout)
    result.validate(cfg)
    if not torch.equal(dones, (success | result.physical_failure | result.timeout)):
        raise RuntimeError("invalid HER terminal accounting")
    return result


def transition_done(batch: TransitionBatch, cfg: StagePPOFDConfig) -> torch.Tensor:
    return (goal_success(batch.next_achieved_goal, batch.desired_goal, batch.task_valid, cfg)
            | batch.physical_failure | batch.timeout)


class ActorMLP(nn.Module):
    """MLP layout matching the current rsl_rl StagePPO actor exactly."""

    def __init__(self, cfg: StagePPOFDConfig):
        super().__init__()
        cfg.validate()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.observation_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, cfg.action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(obs))


class CriticMLP(nn.Module):
    def __init__(self, cfg: StagePPOFDConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.observation_dim + cfg.action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1)).squeeze(-1)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_value, source_value in zip(target.parameters(), source.parameters(), strict=True):
            target_value.lerp_(source_value, tau)


class DDPGfDPretrainer:
    """Off-policy actor warm-start; PPO itself remains unchanged and on-policy."""

    def __init__(self, cfg: StagePPOFDConfig, *, device: str | torch.device = "cpu"):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = ActorMLP(cfg).to(self.device)
        self.critic = CriticMLP(cfg).to(self.device)
        self.target_actor = ActorMLP(cfg).to(self.device)
        self.target_critic = CriticMLP(cfg).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _to_device(self, batch: TransitionBatch) -> TransitionBatch:
        return replace(batch, **{
            name: getattr(batch, name).to(self.device) for name in batch.__dataclass_fields__
        })

    def behavior_clone(self, demonstrations: TransitionBatch) -> dict[str, float]:
        """Warm the same actor MLP from successful demonstrations only."""
        demonstrations.validate(self.cfg)
        demonstrations = self._to_device(demonstrations)
        predicted = self.actor(demonstrations.obs)
        bc_loss = (predicted - demonstrations.actions).square().mean()
        action_l2 = predicted.square().mean()
        loss = bc_loss + self.cfg.action_l2_weight * action_l2
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        # DDPG starts from the BC actor, not the random pre-BC target actor.
        self.target_actor.load_state_dict(self.actor.state_dict())
        return {"bc_loss": float(bc_loss.detach()), "bc_total_loss": float(loss.detach())}

    def update(self, replay: TransitionBatch, demonstrations: TransitionBatch) -> dict[str, float | int]:
        """One DDPG update with a separate demonstration mini-batch."""
        replay.validate(self.cfg)
        demonstrations.validate(self.cfg)
        replay, demonstrations = self._to_device(replay), self._to_device(demonstrations)
        combined = TransitionBatch.concatenate((replay, demonstrations))

        with torch.no_grad():
            next_action = self.target_actor(combined.next_obs)
            target_q = self.target_critic(combined.next_obs, next_action)
            done = transition_done(combined, self.cfg)
            target = combined.rewards + self.cfg.gamma * (~done).float() * target_q
        q = self.critic(combined.obs, combined.actions)
        critic_loss = torch.nn.functional.mse_loss(q, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_action = self.actor(combined.obs)
        policy_loss = -self.critic(combined.obs, policy_action).mean()
        demo_action = self.actor(demonstrations.obs)
        with torch.no_grad():
            expert_q = self.critic(demonstrations.obs, demonstrations.actions)
            actor_q = self.critic(demonstrations.obs, demo_action)
            q_filter = expert_q > actor_q
        per_demo_bc = (demo_action - demonstrations.actions).square().mean(dim=-1)
        bc_loss = per_demo_bc[q_filter].mean() if q_filter.any() else per_demo_bc.sum() * 0
        action_l2 = policy_action.square().mean()
        actor_loss = policy_loss + self.cfg.bc_weight * bc_loss + self.cfg.action_l2_weight * action_l2
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        _soft_update(self.target_actor, self.actor, self.cfg.tau)
        _soft_update(self.target_critic, self.critic, self.cfg.tau)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "bc_loss": float(bc_loss.detach()),
            "q_filter_kept": int(q_filter.sum()),
            "demo_batch": len(demonstrations),
            "replay_batch": len(replay),
        }


def copy_actor_mlp_to_stageppo(source: ActorMLP, stageppo_actor: nn.Module) -> list[str]:
    """Copy only MLP tensors; leave PPO distribution state and critic untouched."""
    source_state = source.state_dict()
    target_state = stageppo_actor.state_dict()
    copied = []
    for name, value in source_state.items():
        if name not in target_state:
            raise ValueError(f"StagePPO actor is missing {name}")
        if target_state[name].shape != value.shape:
            raise ValueError(f"shape mismatch for {name}: {target_state[name].shape} != {value.shape}")
        target_state[name] = value.detach().to(device=target_state[name].device,
                                               dtype=target_state[name].dtype).clone()
        copied.append(name)
    extra_mlp = sorted(name for name in target_state if name.startswith("mlp.") and name not in source_state)
    if extra_mlp:
        raise ValueError(f"unexpected StagePPO MLP tensors: {extra_mlp}")
    stageppo_actor.load_state_dict(target_state, strict=True)
    return copied
