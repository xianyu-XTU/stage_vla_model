"""RSL-RL 5.0.1 PPO variant for M10-12-F1 factor-local terminal credit.

Only the clipped actor surrogate differs from stock PPO.  On samples whose
*pre-action* state was base release-ready, the PPO likelihood ratio is computed
from the GRIP categorical factor only.  All other samples use the original
joint four-factor likelihood.  Critic targets, scalar T1 reward, joint entropy,
adaptive joint KL, optimizer and action execution remain unchanged.

The pre-action release-ready mask is passed from the M10 wrapper through the
standard ``extras`` dictionary and is stored as an additional rollout
``distribution_param`` so RSL-RL's stock shuffled mini-batch generator keeps it
aligned with observations/actions without modifying RSL-RL itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from .action_dsl import M9B_CATEGORY_COUNTS
from .m10_factor_local_credit import (
    M10_F1_READY_EXTRAS_KEY,
    clipped_factor_local_surrogate_loss,
)


class M10FactorLocalGripPPO(PPO):
    """Stock PPO with GRIP-only direct actor credit on terminal ready samples."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.rnd is not None:
            raise RuntimeError("M10-12-F1 does not support RND; M10 config must keep rnd_cfg=None")
        if self.symmetry is not None:
            raise RuntimeError("M10-12-F1 does not support symmetry; M10 config must keep symmetry_cfg=None")
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise RuntimeError("M10-12-F1 currently supports the frozen feed-forward MLP actor/critic only")

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        """Attach the pre-action ready mask to the transition before stock storage."""
        if M10_F1_READY_EXTRAS_KEY not in extras:
            raise KeyError(
                f"M10-12-F1 requires extras['{M10_F1_READY_EXTRAS_KEY}'] from the M10 wrapper"
            )
        if self.transition.distribution_params is None:
            raise RuntimeError("actor distribution params are missing before process_env_step")

        ready = torch.as_tensor(extras[M10_F1_READY_EXTRAS_KEY], device=self.device)
        ready = ready.reshape(-1)
        if ready.shape[0] != rewards.reshape(-1).shape[0]:
            raise ValueError(
                f"release-ready mask batch {ready.shape[0]} != reward batch {rewards.reshape(-1).shape[0]}"
            )
        if ready.dtype != torch.bool:
            raise TypeError(f"{M10_F1_READY_EXTRAS_KEY} must be bool, got {ready.dtype}")
        ready_param = ready.to(dtype=torch.float32).unsqueeze(-1).detach()
        self.transition.distribution_params = (*self.transition.distribution_params, ready_param)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Run stock PPO update with a hybrid joint/GRIP-only actor ratio."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_ready_fraction = 0.0
        mean_ready_surrogate = 0.0
        mean_nonready_surrogate = 0.0
        ready_batches = 0
        nonready_batches = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Recompute the current categorical distribution exactly as stock PPO.
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            current_params = self.actor.output_distribution_params
            if len(current_params) != 1:
                raise RuntimeError(
                    f"M10-12-F1 expects one concatenated categorical-logit tensor, got {len(current_params)} params"
                )
            if batch.old_distribution_params is None or len(batch.old_distribution_params) != 2:
                raise RuntimeError(
                    "M10-12-F1 rollout must contain old logits + pre-action release-ready mask"
                )
            current_logits = current_params[0]
            old_logits = batch.old_distribution_params[0]
            release_ready = batch.old_distribution_params[1]
            entropy = self.actor.output_entropy

            # Keep the same *joint* adaptive-KL trust region as T1.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence((old_logits,), (current_logits,))
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            surrogate_loss, selection = clipped_factor_local_surrogate_loss(
                current_logits,
                old_logits,
                batch.actions,
                batch.advantages,
                release_ready,
                clip_param=self.clip_param,
                category_counts=M9B_CATEGORY_COUNTS,
            )

            # Value function loss is unchanged from stock PPO.
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            ready = selection.release_ready
            mean_ready_fraction += ready.float().mean().item()

            # Read-only diagnostics using the exact selected PPO objective terms.
            with torch.no_grad():
                adv = batch.advantages.squeeze(-1)
                ratio = torch.exp(selection.current_log_prob - selection.old_log_prob)
                s1 = -adv * ratio
                s2 = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                sample_surrogate = torch.max(s1, s2)
                if bool(ready.any().item()):
                    mean_ready_surrogate += sample_surrogate[ready].mean().item()
                    ready_batches += 1
                if bool((~ready).any().item()):
                    mean_nonready_surrogate += sample_surrogate[~ready].mean().item()
                    nonready_batches += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_ready_fraction /= num_updates
        if ready_batches:
            mean_ready_surrogate /= ready_batches
        if nonready_batches:
            mean_nonready_surrogate /= nonready_batches

        self.storage.clear()
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "factor_local_ready_fraction": mean_ready_fraction,
            "factor_local_ready_grip_surrogate": mean_ready_surrogate,
            "factor_local_nonready_joint_surrogate": mean_nonready_surrogate,
        }
