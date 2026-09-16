"""M18 constrained RL-learned RELEASE timing utilities.

The validated A1_RF150 movement residual is executed by a frozen deterministic
policy. A separate PPO actor has exactly one action: ``release_signal``.
Execution thresholds that scalar into KEEP_CLOSED / OPEN, but only after the
causal GeometryPhaseTracker has entered RELEASE. OPEN is then latched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from stage_vla.lightweight_vla.geometry_phase import RELEASE

LEARNED_RELEASE_ACTION_DIM = 1


@dataclass(frozen=True)
class LearnedReleaseConfig:
    threshold: float = 0.0
    wait_free_steps: int = 2
    wait_penalty: float = 0.004
    invalid_request_penalty: float = 0.002

    def validate(self) -> None:
        if self.wait_free_steps < 0:
            raise ValueError("wait_free_steps must be >= 0")
        if self.wait_penalty < 0 or self.invalid_request_penalty < 0:
            raise ValueError("release reward penalties must be >= 0")


class LearnedReleaseController:
    """Vectorized binary OPEN latch driven by one continuous PPO signal."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg: LearnedReleaseConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self.opened = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.release_wait = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.first_open_step = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.opened.zero_()
            self.release_wait.zero_()
            self.first_open_step.fill_(-1)
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.opened[ids] = False
        self.release_wait[ids] = 0
        self.first_open_step[ids] = -1

    def update(self, phase: torch.Tensor, release_signal: torch.Tensor, episode_step: torch.Tensor | None = None):
        phase = phase.to(self.device)
        signal = release_signal.to(self.device)
        eligible = phase == RELEASE
        requested = signal > float(self.cfg.threshold)
        invalid_request = requested & ~eligible & ~self.opened
        opened_now = eligible & requested & ~self.opened
        self.opened |= opened_now
        waiting = eligible & ~self.opened
        self.release_wait = torch.where(
            waiting,
            self.release_wait + 1,
            torch.where(eligible, self.release_wait, torch.zeros_like(self.release_wait)),
        )
        if episode_step is not None:
            step = episode_step.to(self.device, dtype=torch.long)
            self.first_open_step = torch.where(opened_now, step, self.first_open_step)
        grip_cmd = torch.where(
            self.opened,
            torch.ones_like(signal, dtype=torch.float32),
            -torch.ones_like(signal, dtype=torch.float32),
        )
        return grip_cmd, opened_now, invalid_request, waiting

    def reward_adjustment(self, invalid_request: torch.Tensor, waiting: torch.Tensor) -> torch.Tensor:
        excess_wait = waiting & (self.release_wait > int(self.cfg.wait_free_steps))
        return (
            -float(self.cfg.wait_penalty) * excess_wait.float()
            -float(self.cfg.invalid_request_penalty) * invalid_request.float()
        )


class FrozenA1MovementPolicy(nn.Module):
    """Deterministic RSL-RL MLP mean reconstructed from A1 actor_state_dict.

    A1 uses obs_normalization=False and ELU MLP 31->256->128->64->4.  To avoid
    coupling M18 to an extra RSL runner, the deterministic mean is reconstructed
    directly from the saved ``mlp.*`` tensors. The Gaussian std is intentionally
    ignored: deployment/evaluation of A1 also uses the deterministic mean.
    """

    def __init__(self, checkpoint: str | Path, device: torch.device | str):
        super().__init__()
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        actor = ck.get("actor_state_dict")
        if not isinstance(actor, dict):
            raise KeyError("A1 checkpoint missing actor_state_dict")
        weight_keys = sorted(
            [k for k, v in actor.items() if k.startswith("mlp.") and k.endswith(".weight") and torch.is_tensor(v)],
            key=lambda k: int(k.split(".")[1]),
        )
        if not weight_keys:
            raise ValueError("A1 actor has no mlp.*.weight tensors")
        layers = []
        for i, w_key in enumerate(weight_keys):
            w = actor[w_key]
            b_key = w_key[:-6] + "bias"
            b = actor.get(b_key)
            if not torch.is_tensor(b):
                raise KeyError(f"missing {b_key}")
            linear = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                linear.weight.copy_(w)
                linear.bias.copy_(b)
            layers.append(linear)
            if i != len(weight_keys) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers).to(device).eval()
        self.net.requires_grad_(False)
        self.obs_dim = int(actor[weight_keys[0]].shape[1])
        self.action_dim = int(actor[weight_keys[-1]].shape[0])
        if self.action_dim != 4:
            raise ValueError(f"A1 movement actor must output 4 residuals, got {self.action_dim}")

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"A1 movement obs expected {self.obs_dim}, got {obs.shape[-1]}")
        return self.net(obs)


def apply_learned_release_reward_contract(
    base_reward: torch.Tensor,
    *,
    waiting: torch.Tensor,
    release_wait: torch.Tensor,
    invalid_request: torch.Tensor,
    cfg: LearnedReleaseConfig,
) -> torch.Tensor:
    """Prevent RELEASE-wait reward farming while preserving normal PLACE reward.

    During RELEASE while still CLOSED, dense near-stack reward is replaced by
    zero for ``wait_free_steps`` and then a small negative step cost. Invalid
    OPEN requests outside RELEASE receive a tiny penalty.
    """
    out = base_reward - float(cfg.invalid_request_penalty) * invalid_request.float()
    wait_cost = torch.where(
        release_wait > int(cfg.wait_free_steps),
        -torch.full_like(base_reward, float(cfg.wait_penalty)),
        torch.zeros_like(base_reward),
    )
    return torch.where(waiting, wait_cost, out)
