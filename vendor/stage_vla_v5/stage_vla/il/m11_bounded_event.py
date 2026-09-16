"""M11-B-D6 bounded residual estimator and decoupled causal GRIP events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from stage_vla.rl.action_dsl import (
    M9B_CATEGORY_COUNTS,
    M9B_CLOSE_TOKEN,
    M9B_KEEP_TOKEN,
    M9B_OPEN_TOKEN,
)
from stage_vla.rl.factorized_categorical_core import split_factor_logits
from stage_vla.rl.m10_stage_core import M10_STAGE_COUNT, stage_one_hot

from .m11_hybrid_policy import (
    M11HybridMetrics,
    M11HybridModelConfig,
    M11HybridTokenActor,
    grip_release_ready_contract,
    hybrid_metrics,
    load_m11_d4_actor,
    translation_tokens,
)
from .m11_intent_codec import M11IntentDatasetBundle, M11IntentSplit
from .m11_residual_latch import M11CausalFilterReport

M11_D6_CHECKPOINT_SCHEMA = "stage_vla.m11.bounded_residual_decoupled_event_actor.v1"
M11_D6_SUMMARY_SCHEMA = "stage_vla.m11.bounded_residual_decoupled_event_summary.v1"
M11_D6_CONTROLLER_SCHEMA = "stage_vla.m11.close_hazard_automatic_ready_open.v1"
M11_D6_RESIDUAL_SCHEMA = "stage_vla.m11.causal_bounded_residual_estimator.v1"
M11_D8_CHECKPOINT_SCHEMA = (
    "stage_vla.m11.bounded_residual_temporal_confirmation_actor.v1"
)
M11_D8_SUMMARY_SCHEMA = (
    "stage_vla.m11.bounded_residual_temporal_confirmation_summary.v1"
)
M11_D8_CONTROLLER_SCHEMA = (
    "stage_vla.m11.consecutive_close_confirmation_automatic_ready_open.v1"
)


@dataclass(frozen=True)
class M11BoundedEventModelConfig:
    observation_dim: int = 99
    observation_hidden_dims: tuple[int, ...] = (256, 256)
    recurrent_hidden_dim: int = 256
    head_hidden_dim: int = 128
    target_translation_scale_m: float = 0.004
    category_counts: tuple[int, ...] = M9B_CATEGORY_COUNTS
    residual_dim: int = 3
    stage_count: int = M10_STAGE_COUNT
    maximum_abs_residual_estimate_normalized: float = 0.25
    close_adapter_hidden_dim: int = 64
    activation: str = "elu"
    observation_normalization: bool = False

    def __post_init__(self) -> None:
        M11HybridModelConfig(
            observation_dim=self.observation_dim,
            observation_hidden_dims=self.observation_hidden_dims,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            target_translation_scale_m=self.target_translation_scale_m,
            category_counts=self.category_counts,
            activation=self.activation,
            observation_normalization=self.observation_normalization,
        )
        if self.residual_dim != 3:
            raise ValueError("D6 requires one bounded estimate for each XYZ factor")
        if self.stage_count != M10_STAGE_COUNT:
            raise ValueError("D6 requires the unchanged five-stage M10 contract")
        if not 0.0 < self.maximum_abs_residual_estimate_normalized <= 0.25:
            raise ValueError("D6 residual estimate bound must be inside (0,0.25]")
        if self.close_adapter_hidden_dim <= 0:
            raise ValueError("close_adapter_hidden_dim must be > 0")


class M11BoundedEventActor(nn.Module):
    """D4 backbone with a non-accumulating residual estimate and CLOSE score."""

    def __init__(self, config: M11BoundedEventModelConfig) -> None:
        super().__init__()
        self.config = config
        widths = (config.observation_dim, *config.observation_hidden_dims)
        encoder: list[nn.Module] = []
        for index in range(len(widths) - 1):
            encoder.extend((nn.Linear(widths[index], widths[index + 1]), nn.ELU()))
        self.observation_encoder = nn.Sequential(*encoder)
        self.gru = nn.GRU(
            input_size=widths[-1],
            hidden_size=config.recurrent_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(config.recurrent_hidden_dim, config.head_hidden_dim),
            nn.ELU(),
        )
        self.residual_estimator = nn.Linear(config.head_hidden_dim, config.residual_dim)
        self.translation_token_head = nn.Linear(
            config.head_hidden_dim + config.residual_dim, 15
        )
        self.intent_head = nn.Linear(config.head_hidden_dim, 3)
        close_context_dim = config.head_hidden_dim + config.stage_count + 1
        self.close_base = nn.Linear(close_context_dim, 1)
        self.close_adapter = nn.Sequential(
            nn.Linear(close_context_dim, config.close_adapter_hidden_dim),
            nn.ELU(),
            nn.Linear(config.close_adapter_hidden_dim, 1),
        )

    def encode_sequence(
        self,
        observations: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 3 or value.shape[-1] != self.config.observation_dim:
            raise ValueError("observations must have shape [B,T,99]")
        if not torch.isfinite(value).all():
            raise ValueError("observations contain NaN/Inf")
        encoded = self.observation_encoder(value)
        recurrent, next_hidden = self.gru(encoded, hidden)
        return self.shared_head(recurrent), next_hidden

    def close_context(
        self,
        latent: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
    ) -> Tensor:
        stages = torch.as_tensor(stage_ids, device=latent.device, dtype=torch.long)
        ready = torch.as_tensor(release_ready, device=latent.device, dtype=torch.bool)
        if stages.shape != latent.shape[:-1] or ready.shape != latent.shape[:-1]:
            raise ValueError("stage_ids/release_ready must align with latent [B,T]")
        stage_context = stage_one_hot(stages, dtype=latent.dtype).to(latent.device)
        return torch.cat(
            (latent, stage_context, ready.to(latent.dtype).unsqueeze(-1)), dim=-1
        )

    def close_score_from_context(self, context: Tensor) -> Tensor:
        value = torch.as_tensor(context)
        expected = self.config.head_hidden_dim + self.config.stage_count + 1
        if value.shape[-1] != expected:
            raise ValueError(f"close context final dimension must be {expected}")
        return (self.close_base(value) + self.close_adapter(value)).squeeze(-1)

    def decision_heads(
        self,
        latent: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = torch.tanh(self.residual_estimator(latent)) * float(
            self.config.maximum_abs_residual_estimate_normalized
        )
        translation_logits = self.translation_token_head(
            torch.cat((latent, residual), dim=-1)
        )
        intent = torch.tanh(self.intent_head(latent))
        close_score = self.close_score_from_context(
            self.close_context(latent, stage_ids, release_ready)
        )
        return translation_logits, intent, residual, close_score

    def forward_sequence(
        self,
        observations: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        latent, next_hidden = self.encode_sequence(observations, hidden)
        xyz, intent, residual, close = self.decision_heads(
            latent, stage_ids, release_ready
        )
        return xyz, intent, residual, close, next_hidden

    def step(
        self,
        observations: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 2:
            raise ValueError("step expects observations [N,99]")
        xyz, intent, residual, close, next_hidden = self.forward_sequence(
            value.unsqueeze(1),
            torch.as_tensor(stage_ids).unsqueeze(1),
            torch.as_tensor(release_ready).unsqueeze(1),
            hidden,
        )
        return (
            xyz[:, 0],
            intent[:, 0],
            residual[:, 0],
            close[:, 0],
            next_hidden,
        )


def initialize_d6_from_d4(
    actor: M11BoundedEventActor,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    teacher, payload = load_m11_d4_actor(checkpoint, map_location=map_location)
    expected = M11HybridModelConfig(
        observation_dim=actor.config.observation_dim,
        observation_hidden_dims=actor.config.observation_hidden_dims,
        recurrent_hidden_dim=actor.config.recurrent_hidden_dim,
        head_hidden_dim=actor.config.head_hidden_dim,
        target_translation_scale_m=actor.config.target_translation_scale_m,
        category_counts=actor.config.category_counts,
        activation=actor.config.activation,
        observation_normalization=actor.config.observation_normalization,
    )
    if asdict(teacher.config) != asdict(expected):
        raise RuntimeError("D4 checkpoint architecture does not match D6 backbone")
    actor.observation_encoder.load_state_dict(
        teacher.observation_encoder.state_dict(), strict=True
    )
    actor.gru.load_state_dict(teacher.gru.state_dict(), strict=True)
    actor.shared_head.load_state_dict(teacher.shared_head.state_dict(), strict=True)
    actor.intent_head.load_state_dict(teacher.intent_head.state_dict(), strict=True)
    with torch.no_grad():
        actor.residual_estimator.weight.zero_()
        actor.residual_estimator.bias.zero_()
        actor.translation_token_head.weight.zero_()
        actor.translation_token_head.weight[:, : actor.config.head_hidden_dim].copy_(
            teacher.translation_token_head.weight
        )
        actor.translation_token_head.bias.copy_(teacher.translation_token_head.bias)
        actor.close_base.weight.zero_()
        actor.close_base.weight[:, : actor.config.head_hidden_dim].copy_(
            (
                teacher.grip_head.weight[M9B_CLOSE_TOKEN]
                - teacher.grip_head.weight[M9B_KEEP_TOKEN]
            ).unsqueeze(0)
        )
        actor.close_base.bias.copy_(
            (
                teacher.grip_head.bias[M9B_CLOSE_TOKEN]
                - teacher.grip_head.bias[M9B_KEEP_TOKEN]
            ).reshape_as(actor.close_base.bias)
        )
        final = actor.close_adapter[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("D6 close adapter final layer contract changed")
        final.weight.zero_()
        final.bias.zero_()
    return payload


class M11CausalEventController:
    """One learned CLOSE hazard and deterministic pre-action-ready OPEN."""

    OPEN_STATE = 0
    CLOSED_STATE = 1
    RELEASED_STATE = 2

    def __init__(
        self,
        num_envs: int,
        *,
        allowed_close_stage_ids: Sequence[int],
        close_score_threshold: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        allowed = tuple(sorted({int(value) for value in allowed_close_stage_ids}))
        if not allowed or any(value < 0 or value >= M10_STAGE_COUNT for value in allowed):
            raise ValueError("allowed_close_stage_ids must contain valid M10 stages")
        if not math.isfinite(float(close_score_threshold)):
            raise ValueError("close_score_threshold must be finite")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.allowed_close_stage_ids = allowed
        self.close_score_threshold = float(close_score_threshold)
        self.state = torch.full(
            (self.num_envs,), self.OPEN_STATE, dtype=torch.long, device=self.device
        )
        self.close_emissions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.open_emissions = torch.zeros_like(self.close_emissions)
        self.threshold_requests = torch.zeros_like(self.close_emissions)
        self.blocked_wrong_stage_requests = torch.zeros_like(self.close_emissions)

    def reset(self, mask: Tensor | None = None) -> None:
        selected = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if selected.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [num_envs]")
        self.state[selected] = self.OPEN_STATE
        self.close_emissions[selected] = 0
        self.open_emissions[selected] = 0
        self.threshold_requests[selected] = 0
        self.blocked_wrong_stage_requests[selected] = 0

    def step(
        self,
        close_score: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        score = torch.as_tensor(
            close_score, dtype=torch.float32, device=self.device
        ).reshape(-1)
        stages = torch.as_tensor(stage_ids, dtype=torch.long, device=self.device).reshape(-1)
        ready = torch.as_tensor(release_ready, dtype=torch.bool, device=self.device).reshape(-1)
        active = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        expected = (self.num_envs,)
        if score.shape != expected or stages.shape != expected:
            raise ValueError("score/stage must have shape [num_envs]")
        if ready.shape != expected or active.shape != expected:
            raise ValueError("ready/active must have shape [num_envs]")
        if not torch.isfinite(score).all():
            raise ValueError("close score contains NaN/Inf")
        allowed_stage = torch.zeros_like(active)
        for stage_id in self.allowed_close_stage_ids:
            allowed_stage |= stages == int(stage_id)
        requests = active & (score >= self.close_score_threshold)
        self.threshold_requests += requests.to(torch.long)
        blocked = requests & (self.state == self.OPEN_STATE) & ~allowed_stage
        self.blocked_wrong_stage_requests += blocked.to(torch.long)
        was_closed = self.state == self.CLOSED_STATE
        close = (
            requests
            & (self.state == self.OPEN_STATE)
            & allowed_stage
        )
        self.state[close] = self.CLOSED_STATE
        # OPEN is no longer a learned rare class. The unchanged F1 pre-action
        # release-ready truth is the complete causal event trigger.
        open_command = (
            active
            & was_closed
            & ready
        )
        tokens = torch.full(
            (self.num_envs,), M9B_KEEP_TOKEN, dtype=torch.long, device=self.device
        )
        tokens[close] = M9B_CLOSE_TOKEN
        tokens[open_command] = M9B_OPEN_TOKEN
        self.state[open_command] = self.RELEASED_STATE
        self.close_emissions += close.to(torch.long)
        self.open_emissions += open_command.to(torch.long)
        execution_logits = torch.full(
            (self.num_envs, 3), -20.0, dtype=score.dtype, device=self.device
        )
        execution_logits.scatter_(1, tokens.unsqueeze(-1), 20.0)
        return tokens, execution_logits

    def summary(self) -> dict[str, Any]:
        return {
            "schema": M11_D6_CONTROLLER_SCHEMA,
            "environments": self.num_envs,
            "allowed_close_stage_ids": list(self.allowed_close_stage_ids),
            "close_score_threshold": self.close_score_threshold,
            "automatic_open_on_pre_action_ready": True,
            "closed_state_environments": int((self.state >= self.CLOSED_STATE).sum().item()),
            "released_state_environments": int((self.state == self.RELEASED_STATE).sum().item()),
            "close_emissions": int(self.close_emissions.sum().item()),
            "open_emissions": int(self.open_emissions.sum().item()),
            "threshold_requests": int(self.threshold_requests.sum().item()),
            "blocked_wrong_stage_requests": int(
                self.blocked_wrong_stage_requests.sum().item()
            ),
            "maximum_close_emissions_per_environment": int(
                self.close_emissions.max().item()
            ),
            "maximum_open_emissions_per_environment": int(
                self.open_emissions.max().item()
            ),
        }


class M11TemporalConfirmationController:
    """Causal K-frame CLOSE confirmation with deterministic ready-gated OPEN.

    A local-maximum decoder would need a future score to confirm that the peak
    has passed.  This controller is strictly causal: only the current score and
    a per-environment consecutive-request counter are used for the current
    action.
    """

    OPEN_STATE = 0
    CLOSED_STATE = 1
    RELEASED_STATE = 2

    def __init__(
        self,
        num_envs: int,
        *,
        allowed_close_stage_ids: Sequence[int],
        confirmation_steps: int,
        close_score_threshold: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        allowed = tuple(sorted({int(value) for value in allowed_close_stage_ids}))
        if not allowed or any(value < 0 or value >= M10_STAGE_COUNT for value in allowed):
            raise ValueError("allowed_close_stage_ids must contain valid M10 stages")
        if int(confirmation_steps) < 2:
            raise ValueError("D8 confirmation_steps must be >= 2")
        if not math.isfinite(float(close_score_threshold)):
            raise ValueError("close_score_threshold must be finite")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.allowed_close_stage_ids = allowed
        self.confirmation_steps = int(confirmation_steps)
        self.close_score_threshold = float(close_score_threshold)
        self.state = torch.full(
            (self.num_envs,), self.OPEN_STATE, dtype=torch.long, device=self.device
        )
        self.confirmation_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.maximum_confirmation_run = torch.zeros_like(self.confirmation_count)
        self.close_emissions = torch.zeros_like(self.confirmation_count)
        self.open_emissions = torch.zeros_like(self.confirmation_count)
        self.threshold_requests = torch.zeros_like(self.confirmation_count)
        self.unconfirmed_positive_frames = torch.zeros_like(self.confirmation_count)
        self.confirmation_resets = torch.zeros_like(self.confirmation_count)
        self.blocked_wrong_stage_requests = torch.zeros_like(self.confirmation_count)

    def reset(self, mask: Tensor | None = None) -> None:
        selected = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if selected.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [num_envs]")
        self.state[selected] = self.OPEN_STATE
        self.confirmation_count[selected] = 0
        self.maximum_confirmation_run[selected] = 0
        self.close_emissions[selected] = 0
        self.open_emissions[selected] = 0
        self.threshold_requests[selected] = 0
        self.unconfirmed_positive_frames[selected] = 0
        self.confirmation_resets[selected] = 0
        self.blocked_wrong_stage_requests[selected] = 0

    def step(
        self,
        close_score: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        score = torch.as_tensor(
            close_score, dtype=torch.float32, device=self.device
        ).reshape(-1)
        stages = torch.as_tensor(
            stage_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        ready = torch.as_tensor(
            release_ready, dtype=torch.bool, device=self.device
        ).reshape(-1)
        active = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if active_mask is None
            else torch.as_tensor(
                active_mask, dtype=torch.bool, device=self.device
            ).reshape(-1)
        )
        expected = (self.num_envs,)
        if score.shape != expected or stages.shape != expected:
            raise ValueError("score/stage must have shape [num_envs]")
        if ready.shape != expected or active.shape != expected:
            raise ValueError("ready/active must have shape [num_envs]")
        if not torch.isfinite(score).all():
            raise ValueError("close score contains NaN/Inf")

        allowed_stage = torch.zeros_like(active)
        for stage_id in self.allowed_close_stage_ids:
            allowed_stage |= stages == int(stage_id)
        requests = active & (score >= self.close_score_threshold)
        open_state = self.state == self.OPEN_STATE
        qualifying = requests & open_state & allowed_stage
        blocked = requests & open_state & ~allowed_stage
        self.threshold_requests += requests.to(torch.long)
        self.blocked_wrong_stage_requests += blocked.to(torch.long)

        reset_confirmation = active & open_state & ~qualifying
        reset_after_evidence = reset_confirmation & (self.confirmation_count > 0)
        self.confirmation_resets += reset_after_evidence.to(torch.long)
        self.confirmation_count[reset_confirmation] = 0
        self.confirmation_count[qualifying] += 1
        self.maximum_confirmation_run = torch.maximum(
            self.maximum_confirmation_run, self.confirmation_count
        )
        close = qualifying & (
            self.confirmation_count >= self.confirmation_steps
        )
        self.unconfirmed_positive_frames += (qualifying & ~close).to(torch.long)

        was_closed = self.state == self.CLOSED_STATE
        self.state[close] = self.CLOSED_STATE
        self.confirmation_count[close] = 0
        open_command = active & was_closed & ready

        tokens = torch.full(
            (self.num_envs,), M9B_KEEP_TOKEN, dtype=torch.long, device=self.device
        )
        tokens[close] = M9B_CLOSE_TOKEN
        tokens[open_command] = M9B_OPEN_TOKEN
        self.state[open_command] = self.RELEASED_STATE
        self.close_emissions += close.to(torch.long)
        self.open_emissions += open_command.to(torch.long)
        execution_logits = torch.full(
            (self.num_envs, 3), -20.0, dtype=score.dtype, device=self.device
        )
        execution_logits.scatter_(1, tokens.unsqueeze(-1), 20.0)
        return tokens, execution_logits

    def summary(self) -> dict[str, Any]:
        return {
            "schema": M11_D8_CONTROLLER_SCHEMA,
            "environments": self.num_envs,
            "allowed_close_stage_ids": list(self.allowed_close_stage_ids),
            "close_score_threshold": self.close_score_threshold,
            "confirmation_mode": "consecutive_threshold_requests",
            "confirmation_steps": self.confirmation_steps,
            "uses_future_score_or_local_peak": False,
            "automatic_open_on_pre_action_ready": True,
            "closed_state_environments": int(
                (self.state >= self.CLOSED_STATE).sum().item()
            ),
            "released_state_environments": int(
                (self.state == self.RELEASED_STATE).sum().item()
            ),
            "close_emissions": int(self.close_emissions.sum().item()),
            "open_emissions": int(self.open_emissions.sum().item()),
            "threshold_requests": int(self.threshold_requests.sum().item()),
            "unconfirmed_positive_frames": int(
                self.unconfirmed_positive_frames.sum().item()
            ),
            "confirmation_resets": int(self.confirmation_resets.sum().item()),
            "blocked_wrong_stage_requests": int(
                self.blocked_wrong_stage_requests.sum().item()
            ),
            "maximum_confirmation_run_observed": int(
                self.maximum_confirmation_run.max().item()
            ),
            "maximum_close_emissions_per_environment": int(
                self.close_emissions.max().item()
            ),
            "maximum_open_emissions_per_environment": int(
                self.open_emissions.max().item()
            ),
        }


def d6_translation_residual_loss(
    translation_logits: Tensor,
    predicted_intent: Tensor,
    predicted_residual: Tensor,
    target_tokens: Tensor,
    target_intent: Tensor,
    target_residual: Tensor,
    mask: Tensor,
    episode_step_ids: Tensor,
    *,
    xyz_class_weights: Sequence[Tensor] | None,
    translation_token_weight: float,
    intent_auxiliary_weight: float,
    residual_supervision_weight: float,
    first_step_weight: float,
    early_step_weight: float,
    early_step_cutoff: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid = torch.as_tensor(mask, dtype=torch.bool, device=translation_logits.device)
    tokens = torch.as_tensor(target_tokens, dtype=torch.long, device=translation_logits.device)
    intent_target = torch.as_tensor(
        target_intent, dtype=predicted_intent.dtype, device=predicted_intent.device
    )
    residual_target = torch.as_tensor(
        target_residual,
        dtype=predicted_residual.dtype,
        device=predicted_residual.device,
    )
    step_ids = torch.as_tensor(
        episode_step_ids, dtype=torch.long, device=translation_logits.device
    )
    if translation_logits.shape != (*valid.shape, 15):
        raise ValueError("translation logits must align as [B,T,15]")
    if tokens.shape != (*valid.shape, 4):
        raise ValueError("target tokens must align as [B,T,4]")
    if predicted_intent.shape != (*valid.shape, 3):
        raise ValueError("predicted intent must align as [B,T,3]")
    if predicted_residual.shape != (*valid.shape, 3):
        raise ValueError("predicted residual must align as [B,T,3]")
    if intent_target.shape != predicted_intent.shape:
        raise ValueError("intent targets are not aligned")
    if residual_target.shape != predicted_residual.shape:
        raise ValueError("residual targets are not aligned")
    if step_ids.shape != valid.shape or not bool(valid.any().item()):
        raise ValueError("episode step ids/mask are invalid")
    if xyz_class_weights is not None and len(xyz_class_weights) != 3:
        raise ValueError("xyz_class_weights must contain three tensors")
    if early_step_cutoff < 1:
        raise ValueError("early_step_cutoff must be >= 1")
    per_step = torch.ones(valid.shape, dtype=translation_logits.dtype, device=valid.device)
    per_step[(step_ids > 0) & (step_ids < int(early_step_cutoff))] = float(
        early_step_weight
    )
    per_step[step_ids == 0] = float(first_step_weight)
    xyz_losses: list[Tensor] = []
    for factor, part in enumerate(split_factor_logits(translation_logits, (5, 5, 5))):
        weight = None
        if xyz_class_weights is not None:
            weight = torch.as_tensor(
                xyz_class_weights[factor], dtype=part.dtype, device=part.device
            )
        ce = F.cross_entropy(
            part.reshape(-1, 5),
            tokens[..., factor].reshape(-1),
            weight=weight,
            reduction="none",
        ).reshape(valid.shape)
        xyz_losses.append((ce[valid] * per_step[valid]).sum() / per_step[valid].sum())
    xyz_loss = torch.stack(xyz_losses).mean()
    intent_loss = F.smooth_l1_loss(predicted_intent[valid], intent_target[valid])
    residual_loss = F.smooth_l1_loss(predicted_residual[valid], residual_target[valid])
    total = (
        float(translation_token_weight) * xyz_loss
        + float(intent_auxiliary_weight) * intent_loss
        + float(residual_supervision_weight) * residual_loss
    )
    return total, {
        "translation_token_ce": xyz_loss,
        "intent_auxiliary": intent_loss,
        "residual_supervision": residual_loss,
    }


def d6_close_event_ranking_loss(
    close_scores: Tensor,
    target_grip_tokens: Tensor,
    stage_ids: Tensor,
    *,
    allowed_close_stage_ids: Sequence[int],
    hard_negative_count: int,
    ranking_margin: float,
    ranking_weight: float,
    confirmation_steps: int = 1,
    confirmation_positive_weight: float = 1.0,
    pre_target_penalty_window: int = 0,
    pre_target_penalty_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    scores = torch.as_tensor(close_scores).reshape(-1)
    target = torch.as_tensor(
        target_grip_tokens, dtype=torch.long, device=scores.device
    ).reshape(-1)
    stages = torch.as_tensor(stage_ids, dtype=torch.long, device=scores.device).reshape(-1)
    if scores.shape != target.shape or scores.shape != stages.shape:
        raise ValueError("close scores, targets and stages must align")
    if not torch.isfinite(scores).all():
        raise ValueError("close scores contain NaN/Inf")
    if int(hard_negative_count) <= 0:
        raise ValueError("hard_negative_count must be > 0")
    if not math.isfinite(ranking_margin) or ranking_margin <= 0.0:
        raise ValueError("ranking_margin must be finite and > 0")
    if not math.isfinite(ranking_weight) or ranking_weight < 0.0:
        raise ValueError("ranking_weight must be finite and >= 0")
    if int(confirmation_steps) <= 0:
        raise ValueError("confirmation_steps must be > 0")
    if (
        not math.isfinite(confirmation_positive_weight)
        or confirmation_positive_weight <= 0.0
    ):
        raise ValueError("confirmation_positive_weight must be finite and > 0")
    if int(pre_target_penalty_window) < 0:
        raise ValueError("pre_target_penalty_window must be >= 0")
    if (
        not math.isfinite(pre_target_penalty_weight)
        or pre_target_penalty_weight < 0.0
    ):
        raise ValueError("pre_target_penalty_weight must be finite and >= 0")
    positives = torch.nonzero(target == M9B_CLOSE_TOKEN, as_tuple=False).flatten()
    if positives.numel() != 1:
        raise ValueError("each event episode must contain exactly one CLOSE target")
    candidate = torch.zeros_like(target, dtype=torch.bool)
    for stage_id in allowed_close_stage_ids:
        candidate |= stages == int(stage_id)
    positive_index = int(positives[0].item())
    if not bool(candidate[positive_index].item()):
        raise ValueError("CLOSE target is outside allowed CLOSE stages")
    positive_start = positive_index - int(confirmation_steps) + 1
    if positive_start < 0:
        raise ValueError("CLOSE target has insufficient confirmation history")
    confirmation_mask = torch.zeros_like(candidate)
    confirmation_mask[positive_start : positive_index + 1] = True
    if not bool(candidate[confirmation_mask].all().item()):
        raise ValueError("confirmation window leaves the allowed CLOSE stage")
    negative = candidate & ~confirmation_mask
    negative_scores = scores[negative]
    if negative_scores.numel() == 0:
        raise ValueError("event episode has no CLOSE-stage negatives")
    k = min(int(hard_negative_count), int(negative_scores.numel()))
    hard = torch.topk(negative_scores, k=k, largest=True).values
    confirmation_scores = scores[confirmation_mask]
    positive_score = scores[positive_index]
    minimum_confirmation_score = confirmation_scores.min()
    positive_loss = F.softplus(-confirmation_scores).mean()
    negative_loss = F.softplus(hard).mean()
    ranking_loss = F.relu(
        float(ranking_margin) - minimum_confirmation_score + hard.max()
    )
    pre_target_start = max(
        0, positive_start - int(pre_target_penalty_window)
    )
    pre_target_mask = candidate.clone()
    pre_target_mask[:pre_target_start] = False
    pre_target_mask[positive_start:] = False
    if bool(pre_target_mask.any().item()):
        # D8 v3: top-k left residual positive frames that formed K-consecutive
        # runs before target (early rate 100%). Penalize EVERY positive prefix
        # frame (hinge-square mean over positives) so no sub-K run survives.
        prefix_positive = F.relu(scores[pre_target_mask]).square()
        prefix_positives = prefix_positive[prefix_positive > 0]
        pre_target_penalty = (
            prefix_positives.mean() if prefix_positives.numel() > 0 else scores.sum() * 0.0
        )
    else:
        pre_target_penalty = scores.sum() * 0.0
    total = (
        float(confirmation_positive_weight) * positive_loss
        + negative_loss
        + float(ranking_weight) * ranking_loss
        + float(pre_target_penalty_weight) * pre_target_penalty
    )
    return total, {
        "positive_softplus": positive_loss,
        "hard_negative_softplus": negative_loss,
        "ranking_margin": ranking_loss,
        "pre_target_hard_hinge_square": pre_target_penalty,
        "positive_score": positive_score.detach(),
        "minimum_confirmation_score": minimum_confirmation_score.detach(),
        "maximum_negative_score": hard.max().detach(),
    }


@dataclass(frozen=True)
class M11ResidualEstimateMetrics:
    num_steps: int
    mae_normalized: tuple[float, float, float]
    mae_mm: tuple[float, float, float]
    rmse_normalized: tuple[float, float, float]
    maximum_abs_prediction_normalized: float


@dataclass(frozen=True)
class M11CloseEventMetrics:
    episodes: int
    target_events: int
    target_above_threshold: int
    target_recall: float
    negative_candidates: int
    false_positive_candidates: int
    false_positive_rate: float
    close_emissions: int
    close_episode_coverage: float
    open_emissions: int
    open_episode_coverage: float
    mean_target_minus_max_negative_margin: float
    minimum_target_minus_max_negative_margin: float
    confirmation_steps: int
    confirmation_window_events: int
    confirmation_window_recall: float
    early_confirmation_episodes: int
    early_confirmation_rate: float
    exact_close_episodes: int
    exact_close_episode_recall: float
    mean_maximum_pre_target_positive_run: float
    maximum_pre_target_positive_run: int


@dataclass(frozen=True)
class M11BoundedEventEvaluation:
    metrics: M11HybridMetrics
    translation_exact_accuracy: float
    residual_metrics: M11ResidualEstimateMetrics
    close_event_metrics: M11CloseEventMetrics
    resolved_grip_state_accuracy: float
    controller_summary: Mapping[str, Any]
    release_ready_contract: Mapping[str, Any]
    event_timing: tuple[Mapping[str, Any], ...]
    episode_ids: tuple[int, ...]
    predicted_tokens_by_episode: tuple[Tensor, ...]
    target_tokens_by_episode: tuple[Tensor, ...]


def _resolved_grip_state(tokens: Tensor) -> Tensor:
    value = torch.as_tensor(tokens, dtype=torch.long).cpu()
    closed = False
    states: list[bool] = []
    for token in value[:, 3].tolist():
        if int(token) == M9B_CLOSE_TOKEN:
            closed = True
        elif int(token) == M9B_OPEN_TOKEN:
            closed = False
        states.append(closed)
    return torch.tensor(states, dtype=torch.bool)


def _first_event(tokens: Tensor, event: int) -> int | None:
    indices = torch.nonzero(tokens[:, 3] == int(event), as_tuple=False).flatten()
    return int(indices[0].item()) if indices.numel() else None


def _maximum_true_run(values: Tensor) -> int:
    run = 0
    maximum = 0
    for value in torch.as_tensor(values, dtype=torch.bool).reshape(-1).tolist():
        run = run + 1 if bool(value) else 0
        maximum = max(maximum, run)
    return maximum


def evaluate_bounded_event_split(
    actor: M11BoundedEventActor,
    split: M11IntentSplit,
    device: str | torch.device,
    *,
    allowed_close_stage_ids: Sequence[int],
    close_score_threshold: float,
    close_confirmation_steps: int = 1,
) -> M11BoundedEventEvaluation:
    if int(close_confirmation_steps) <= 0:
        raise ValueError("close_confirmation_steps must be > 0")
    target_device = torch.device(device)
    episodes: list[tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
    for episode_id in torch.unique(split.episode_ids, sorted=True).tolist():
        selected = split.episode_ids == int(episode_id)
        episodes.append(
            (
                int(episode_id),
                split.observations[selected],
                split.ideal_translation_normalized[selected],
                split.residual_before_normalized[selected],
                split.tokens[selected],
                split.stage_ids[selected],
                split.pre_release_ready[selected],
            )
        )
    if not episodes:
        raise RuntimeError("D6 evaluation split contains no episodes")
    num_episodes = len(episodes)
    max_steps = max(int(item[4].shape[0]) for item in episodes)
    lengths = torch.tensor(
        [item[4].shape[0] for item in episodes], dtype=torch.long, device=target_device
    )
    valid = torch.arange(max_steps, device=target_device).unsqueeze(0) < lengths.unsqueeze(1)
    observations = torch.zeros(
        num_episodes, max_steps, actor.config.observation_dim, device=target_device
    )
    target_intent = torch.zeros(num_episodes, max_steps, 3)
    target_residual = torch.zeros(num_episodes, max_steps, 3)
    target_tokens = torch.zeros(num_episodes, max_steps, 4, dtype=torch.long)
    stages = torch.zeros(num_episodes, max_steps, dtype=torch.long, device=target_device)
    ready = torch.zeros(num_episodes, max_steps, dtype=torch.bool, device=target_device)
    for index, (_id, obs, intent, residual, tokens, stage, pre_ready) in enumerate(episodes):
        length = int(tokens.shape[0])
        observations[index, :length] = obs.to(target_device)
        target_intent[index, :length] = intent
        target_residual[index, :length] = residual
        target_tokens[index, :length] = tokens
        stages[index, :length] = stage.to(target_device)
        ready[index, :length] = pre_ready.to(target_device)

    actor.eval()
    with torch.inference_mode():
        xyz_logits, intent, residual, close_scores, _ = actor.forward_sequence(
            observations, stages, ready
        )
        xyz = translation_tokens(xyz_logits)
        if int(close_confirmation_steps) == 1:
            controller: M11CausalEventController | M11TemporalConfirmationController
            controller = M11CausalEventController(
                num_episodes,
                allowed_close_stage_ids=allowed_close_stage_ids,
                close_score_threshold=close_score_threshold,
                device=target_device,
            )
        else:
            controller = M11TemporalConfirmationController(
                num_episodes,
                allowed_close_stage_ids=allowed_close_stage_ids,
                confirmation_steps=int(close_confirmation_steps),
                close_score_threshold=close_score_threshold,
                device=target_device,
            )
        grip_steps: list[Tensor] = []
        for step in range(max_steps):
            grip, _ = controller.step(
                close_scores[:, step],
                stages[:, step],
                ready[:, step],
                active_mask=valid[:, step],
            )
            grip_steps.append(grip)
        grip_tokens = torch.stack(grip_steps, dim=1)
        predicted = torch.cat((xyz, grip_tokens.unsqueeze(-1)), dim=-1).cpu()

    valid_cpu = valid.cpu()
    target_valid = target_tokens[valid_cpu]
    predicted_valid = predicted[valid_cpu]
    intent_cpu = intent.cpu()
    metrics = hybrid_metrics(
        predicted_valid,
        target_valid,
        intent_cpu[valid_cpu],
        target_intent[valid_cpu],
        target_translation_scale_m=actor.config.target_translation_scale_m,
        close_emissions=int(controller.close_emissions.sum().item()),
        open_emissions=int(controller.open_emissions.sum().item()),
    )
    translation_exact = float(
        (predicted_valid[:, :3] == target_valid[:, :3])
        .all(dim=-1)
        .to(torch.float64)
        .mean()
        .item()
    )
    residual_cpu = residual.cpu()
    residual_error = residual_cpu[valid_cpu] - target_residual[valid_cpu]
    residual_mae = residual_error.abs().mean(dim=0)
    residual_rmse = residual_error.square().mean(dim=0).sqrt()
    residual_metrics = M11ResidualEstimateMetrics(
        num_steps=int(residual_error.shape[0]),
        mae_normalized=tuple(float(value) for value in residual_mae.tolist()),
        mae_mm=tuple(
            float(value) * actor.config.target_translation_scale_m * 1000.0
            for value in residual_mae.tolist()
        ),
        rmse_normalized=tuple(float(value) for value in residual_rmse.tolist()),
        maximum_abs_prediction_normalized=float(
            residual_cpu[valid_cpu].abs().max().item()
        ),
    )

    close_cpu = close_scores.cpu()
    stages_cpu = stages.cpu()
    candidate = torch.zeros_like(valid_cpu)
    for stage_id in allowed_close_stage_ids:
        candidate |= stages_cpu == int(stage_id)
    candidate &= valid_cpu
    target_close = target_tokens[..., 3] == M9B_CLOSE_TOKEN
    positive_at_target = (close_cpu >= float(close_score_threshold)) & target_close
    confirmation_evidence = torch.zeros_like(candidate)
    for index, (_episode_id, _obs, _intent, _residual, tokens, _stage, _ready) in enumerate(episodes):
        close_indices = torch.nonzero(
            tokens[:, 3] == M9B_CLOSE_TOKEN, as_tuple=False
        ).flatten()
        if close_indices.numel() != 1:
            raise RuntimeError("D6/D8 evaluation requires exactly one CLOSE per episode")
        close_index = int(close_indices[0].item())
        confirmation_start = close_index - int(close_confirmation_steps) + 1
        if confirmation_start < 0:
            raise RuntimeError("CLOSE target has insufficient confirmation history")
        confirmation_evidence[index, confirmation_start : close_index + 1] = True
    if not bool((~confirmation_evidence | candidate).all().item()):
        raise RuntimeError("confirmation evidence leaves the allowed CLOSE stage")
    negative = candidate & ~confirmation_evidence
    false_positive = negative & (close_cpu >= float(close_score_threshold))
    margins: list[float] = []
    pre_target_runs: list[int] = []
    predicted_by_episode: list[Tensor] = []
    target_by_episode: list[Tensor] = []
    timing: list[Mapping[str, Any]] = []
    correct_states = 0
    state_steps = 0
    confirmation_window_events = 0
    early_confirmation_episodes = 0
    exact_close_episodes = 0
    for index, (episode_id, _obs, _intent, _residual, tokens, stage, _ready) in enumerate(episodes):
        length = int(tokens.shape[0])
        predicted_episode = predicted[index, :length].clone()
        predicted_by_episode.append(predicted_episode)
        target_by_episode.append(tokens.clone())
        predicted_state = _resolved_grip_state(predicted_episode)
        target_state = _resolved_grip_state(tokens)
        correct_states += int((predicted_state == target_state).sum().item())
        state_steps += length
        close_index = int(
            torch.nonzero(tokens[:, 3] == M9B_CLOSE_TOKEN, as_tuple=False)[0].item()
        )
        confirmation_start = close_index - int(close_confirmation_steps) + 1
        stage_candidates = stage == int(allowed_close_stage_ids[0])
        for stage_id in allowed_close_stage_ids[1:]:
            stage_candidates |= stage == int(stage_id)
        negatives = stage_candidates.clone()
        negatives[confirmation_start : close_index + 1] = False
        negative_values = close_cpu[index, :length][negatives]
        if negative_values.numel() == 0:
            raise RuntimeError("event episode has no CLOSE-stage negative candidates")
        confirmation_values = close_cpu[
            index, confirmation_start : close_index + 1
        ]
        confirmation_passed = bool(
            (confirmation_values >= float(close_score_threshold)).all().item()
        )
        confirmation_window_events += int(confirmation_passed)
        margin = float(
            confirmation_values.min().item() - negative_values.max().item()
        )
        margins.append(margin)
        raw_requests = (
            (close_cpu[index, :length] >= float(close_score_threshold))
            & stage_candidates
        )
        maximum_pre_target_run = _maximum_true_run(raw_requests[:close_index])
        pre_target_runs.append(maximum_pre_target_run)
        raw_indices = torch.nonzero(raw_requests, as_tuple=False).flatten()
        first_raw_threshold = (
            int(raw_indices[0].item()) if raw_indices.numel() else None
        )
        target_open = _first_event(tokens, M9B_OPEN_TOKEN)
        predicted_open = _first_event(predicted_episode, M9B_OPEN_TOKEN)
        predicted_close = _first_event(predicted_episode, M9B_CLOSE_TOKEN)
        if predicted_close is not None and predicted_close < close_index:
            early_confirmation_episodes += 1
        if predicted_close == close_index:
            exact_close_episodes += 1
        timing.append(
            {
                "episode_id": episode_id,
                "target_close_step": close_index,
                "predicted_close_step": predicted_close,
                "close_step_error": (
                    predicted_close - close_index if predicted_close is not None else None
                ),
                "target_open_step": target_open,
                "predicted_open_step": predicted_open,
                "open_step_error": (
                    predicted_open - target_open
                    if predicted_open is not None and target_open is not None
                    else None
                ),
                "target_close_score": float(close_cpu[index, close_index].item()),
                "confirmation_start_step": confirmation_start,
                "confirmation_steps": int(close_confirmation_steps),
                "confirmation_window_all_above_threshold": confirmation_passed,
                "minimum_confirmation_score": float(
                    confirmation_values.min().item()
                ),
                "first_raw_threshold_step": first_raw_threshold,
                "maximum_pre_target_positive_run": maximum_pre_target_run,
                "confirmation_minus_max_negative_margin": margin,
            }
        )
    close_metrics = M11CloseEventMetrics(
        episodes=num_episodes,
        target_events=int(target_close.sum().item()),
        target_above_threshold=int(positive_at_target.sum().item()),
        target_recall=float(positive_at_target.sum().item() / num_episodes),
        negative_candidates=int(negative.sum().item()),
        false_positive_candidates=int(false_positive.sum().item()),
        false_positive_rate=(
            float(false_positive.sum().item() / negative.sum().item())
            if bool(negative.any().item())
            else 0.0
        ),
        close_emissions=int(controller.close_emissions.sum().item()),
        close_episode_coverage=float(controller.close_emissions.sum().item() / num_episodes),
        open_emissions=int(controller.open_emissions.sum().item()),
        open_episode_coverage=float(controller.open_emissions.sum().item() / num_episodes),
        mean_target_minus_max_negative_margin=sum(margins) / len(margins),
        minimum_target_minus_max_negative_margin=min(margins),
        confirmation_steps=int(close_confirmation_steps),
        confirmation_window_events=confirmation_window_events,
        confirmation_window_recall=confirmation_window_events / num_episodes,
        early_confirmation_episodes=early_confirmation_episodes,
        early_confirmation_rate=early_confirmation_episodes / num_episodes,
        exact_close_episodes=exact_close_episodes,
        exact_close_episode_recall=exact_close_episodes / num_episodes,
        mean_maximum_pre_target_positive_run=(
            sum(pre_target_runs) / len(pre_target_runs)
        ),
        maximum_pre_target_positive_run=max(pre_target_runs),
    )
    return M11BoundedEventEvaluation(
        metrics=metrics,
        translation_exact_accuracy=translation_exact,
        residual_metrics=residual_metrics,
        close_event_metrics=close_metrics,
        resolved_grip_state_accuracy=correct_states / state_steps if state_steps else 0.0,
        controller_summary=controller.summary(),
        release_ready_contract=grip_release_ready_contract(split),
        event_timing=tuple(timing),
        episode_ids=tuple(item[0] for item in episodes),
        predicted_tokens_by_episode=tuple(predicted_by_episode),
        target_tokens_by_episode=tuple(target_by_episode),
    )


def d6_translation_selection_score(evaluation: M11BoundedEventEvaluation) -> float:
    metrics = evaluation.metrics
    residual = evaluation.residual_metrics
    return float(
        2.0 * (1.0 - evaluation.translation_exact_accuracy)
        + sum(1.0 - value for value in metrics.factor_accuracy[:3])
        + 0.5 * sum(residual.mae_normalized) / 3.0
        + 0.05 * sum(metrics.translation_intent_mae_mm) / 3.0
    )


def d6_event_selection_score(evaluation: M11BoundedEventEvaluation) -> float:
    event = evaluation.close_event_metrics
    metrics = evaluation.metrics
    mean_margin_penalty = (
        math.log1p(math.exp(-event.mean_target_minus_max_negative_margin))
        if event.mean_target_minus_max_negative_margin >= 0.0
        else -event.mean_target_minus_max_negative_margin
        + math.log1p(math.exp(event.mean_target_minus_max_negative_margin))
    )
    minimum_margin_penalty = (
        math.log1p(math.exp(-event.minimum_target_minus_max_negative_margin))
        if event.minimum_target_minus_max_negative_margin >= 0.0
        else -event.minimum_target_minus_max_negative_margin
        + math.log1p(math.exp(event.minimum_target_minus_max_negative_margin))
    )
    return float(
        3.0 * (1.0 - event.target_recall)
        + 2.0 * event.false_positive_rate
        + (1.0 - event.close_episode_coverage)
        + (1.0 - event.open_episode_coverage)
        + (1.0 - metrics.grip_close_recall)
        + (1.0 - metrics.grip_open_recall)
        + 2.0 * (1.0 - evaluation.resolved_grip_state_accuracy)
        + 0.2 * mean_margin_penalty
        + 0.1 * minimum_margin_penalty
    )


def d8_event_selection_score(evaluation: M11BoundedEventEvaluation) -> float:
    """Prefer exact causal confirmations, not isolated raw score crossings."""

    event = evaluation.close_event_metrics
    metrics = evaluation.metrics
    mean_margin = event.mean_target_minus_max_negative_margin
    minimum_margin = event.minimum_target_minus_max_negative_margin
    mean_margin_penalty = (
        math.log1p(math.exp(-mean_margin))
        if mean_margin >= 0.0
        else -mean_margin + math.log1p(math.exp(mean_margin))
    )
    minimum_margin_penalty = (
        math.log1p(math.exp(-minimum_margin))
        if minimum_margin >= 0.0
        else -minimum_margin + math.log1p(math.exp(minimum_margin))
    )
    return float(
        5.0 * (1.0 - event.exact_close_episode_recall)
        + 4.0 * event.early_confirmation_rate
        + 2.0 * (1.0 - event.confirmation_window_recall)
        + (1.0 - event.close_episode_coverage)
        + (1.0 - event.open_episode_coverage)
        + (1.0 - metrics.grip_close_recall)
        + (1.0 - metrics.grip_open_recall)
        + 2.0 * (1.0 - evaluation.resolved_grip_state_accuracy)
        + 0.2 * mean_margin_penalty
        + 0.1 * minimum_margin_penalty
    )


def d6_checkpoint_payload(
    actor: M11BoundedEventActor,
    *,
    translation_epoch: int,
    event_epoch: int,
    evaluation: M11BoundedEventEvaluation,
    translation_selection_score: float,
    event_selection_score: float,
    dataset: M11IntentDatasetBundle,
    filter_reports: Mapping[str, M11CausalFilterReport],
    training_config: Mapping[str, Any],
    d4_checkpoint: str,
    allowed_close_stage_ids: Sequence[int],
    close_score_threshold: float,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": M11_D6_CHECKPOINT_SCHEMA,
        "translation_epoch": int(translation_epoch),
        "event_epoch": int(event_epoch),
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "validation_metrics": asdict(evaluation.metrics),
        "validation_translation_exact_accuracy": float(
            evaluation.translation_exact_accuracy
        ),
        "validation_residual_metrics": asdict(evaluation.residual_metrics),
        "validation_close_event_metrics": asdict(evaluation.close_event_metrics),
        "validation_resolved_grip_state_accuracy": float(
            evaluation.resolved_grip_state_accuracy
        ),
        "validation_controller_summary": dict(evaluation.controller_summary),
        "translation_selection_score": float(translation_selection_score),
        "event_selection_score": float(event_selection_score),
        "training_config": dict(training_config),
        "initialized_from_d4_checkpoint": str(d4_checkpoint),
        "dataset": {
            "directory": dataset.dataset_dir,
            "manifest_sha256": dataset.manifest_sha256,
            "quantizer_method": dataset.quantizer_method,
            "train_episodes": dataset.train.episode_count,
            "validation_episodes": dataset.validation.episode_count,
            "minimum_episode_steps": min(
                dataset.train.minimum_episode_steps,
                dataset.validation.minimum_episode_steps,
            ),
            "maximum_episode_steps": max(
                dataset.train.maximum_episode_steps,
                dataset.validation.maximum_episode_steps,
            ),
            "causal_filter": {
                name: asdict(report) for name, report in filter_reports.items()
            },
        },
        "residual_estimator_contract": {
            "schema": M11_D6_RESIDUAL_SCHEMA,
            "causal_recurrent_history_only": True,
            "teacher_residual_is_training_target_only": True,
            "runtime_accumulator_used": False,
            "predicted_intent_accumulated": False,
            "maximum_abs_prediction_normalized": float(
                actor.config.maximum_abs_residual_estimate_normalized
            ),
        },
        "controller_contract": {
            "schema": M11_D6_CONTROLLER_SCHEMA,
            "initial_state": "OPEN",
            "allowed_close_stage_ids": [int(value) for value in allowed_close_stage_ids],
            "close_score_threshold": float(close_score_threshold),
            "automatic_open_on_pre_action_ready": True,
            "learned_open_class_used": False,
            "post_action_truth_used_for_current_action": False,
            "maximum_close_emissions_per_episode": 1,
            "maximum_open_emissions_per_episode": 1,
        },
        "training_contract": {
            "translation_and_event_optimization_decoupled": True,
            "event_phase_backbone_frozen": True,
            "close_positive_vs_hard_negative_ranking": True,
            "ppo_authorized": False,
        },
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    return payload


def d8_checkpoint_payload(
    actor: M11BoundedEventActor,
    *,
    translation_epoch: int,
    event_epoch: int,
    evaluation: M11BoundedEventEvaluation,
    translation_selection_score: float,
    event_selection_score: float,
    dataset: M11IntentDatasetBundle,
    filter_reports: Mapping[str, M11CausalFilterReport],
    training_config: Mapping[str, Any],
    d4_checkpoint: str,
    translation_checkpoint: str,
    allowed_close_stage_ids: Sequence[int],
    close_score_threshold: float,
    close_confirmation_steps: int,
    pre_target_penalty_window: int,
    pre_target_penalty_weight: float,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(close_confirmation_steps) < 2:
        raise ValueError("D8 close_confirmation_steps must be >= 2")
    payload = d6_checkpoint_payload(
        actor,
        translation_epoch=translation_epoch,
        event_epoch=event_epoch,
        evaluation=evaluation,
        translation_selection_score=translation_selection_score,
        event_selection_score=event_selection_score,
        dataset=dataset,
        filter_reports=filter_reports,
        training_config=training_config,
        d4_checkpoint=d4_checkpoint,
        allowed_close_stage_ids=allowed_close_stage_ids,
        close_score_threshold=close_score_threshold,
        optimizer_state_dict=optimizer_state_dict,
    )
    payload["schema"] = M11_D8_CHECKPOINT_SCHEMA
    payload["initialized_from_translation_checkpoint"] = str(translation_checkpoint)
    payload["controller_contract"] = {
        "schema": M11_D8_CONTROLLER_SCHEMA,
        "initial_state": "OPEN",
        "allowed_close_stage_ids": [int(value) for value in allowed_close_stage_ids],
        "close_score_threshold": float(close_score_threshold),
        "confirmation_mode": "consecutive_threshold_requests",
        "confirmation_steps": int(close_confirmation_steps),
        "target_evidence_window_ends_at_expert_close": True,
        "uses_future_score_or_local_peak": False,
        "isolated_short_runs_are_suppressed": True,
        "automatic_open_on_pre_action_ready": True,
        "learned_open_class_used": False,
        "post_action_truth_used_for_current_action": False,
        "maximum_close_emissions_per_episode": 1,
        "maximum_open_emissions_per_episode": 1,
    }
    payload["training_contract"] = {
        "translation_and_event_optimization_decoupled": True,
        "translation_checkpoint_reused_without_retraining": True,
        "event_phase_backbone_frozen": True,
        "close_confirmation_window_matches_runtime_counter": True,
        "close_positive_vs_hard_negative_ranking": True,
        "pre_target_hard_hinge_square": True,
        "pre_target_penalty_window": int(pre_target_penalty_window),
        "pre_target_penalty_weight": float(pre_target_penalty_weight),
        "raw_frame_false_positive_is_diagnostic_only": True,
        "ppo_authorized": False,
    }
    return payload


def load_m11_d6_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11BoundedEventActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D6_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D6 checkpoint schema")
    residual = payload.get("residual_estimator_contract", {})
    if residual.get("schema") != M11_D6_RESIDUAL_SCHEMA:
        raise RuntimeError("M11-B-D6 residual estimator contract mismatch")
    if residual.get("causal_recurrent_history_only") is not True:
        raise RuntimeError("M11-B-D6 residual estimator must be causal")
    if residual.get("teacher_residual_is_training_target_only") is not True:
        raise RuntimeError("M11-B-D6 teacher residual must remain training-only")
    if residual.get("runtime_accumulator_used") is not False:
        raise RuntimeError("M11-B-D6 forbids the D5 runtime residual accumulator")
    if residual.get("predicted_intent_accumulated") is not False:
        raise RuntimeError("M11-B-D6 forbids predicted-intent residual accumulation")
    controller = payload.get("controller_contract", {})
    if controller.get("schema") != M11_D6_CONTROLLER_SCHEMA:
        raise RuntimeError("M11-B-D6 controller contract mismatch")
    if controller.get("automatic_open_on_pre_action_ready") is not True:
        raise RuntimeError("M11-B-D6 automatic causal OPEN contract mismatch")
    if controller.get("learned_open_class_used") is not False:
        raise RuntimeError("M11-B-D6 must not restore the sparse learned OPEN class")
    if controller.get("post_action_truth_used_for_current_action") is not False:
        raise RuntimeError("M11-B-D6 forbids post-action truth leakage")
    if controller.get("maximum_close_emissions_per_episode") != 1:
        raise RuntimeError("M11-B-D6 CLOSE one-shot contract mismatch")
    if controller.get("maximum_open_emissions_per_episode") != 1:
        raise RuntimeError("M11-B-D6 OPEN one-shot contract mismatch")
    allowed = controller.get("allowed_close_stage_ids")
    if (
        not isinstance(allowed, (list, tuple))
        or not allowed
        or any(not isinstance(value, int) or value < 0 or value >= M10_STAGE_COUNT for value in allowed)
    ):
        raise RuntimeError("M11-B-D6 CLOSE-stage contract mismatch")
    threshold = controller.get("close_score_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise RuntimeError("M11-B-D6 CLOSE threshold contract mismatch")
    raw = payload["model_config"]
    config = M11BoundedEventModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(int(v) for v in raw["observation_hidden_dims"]),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        target_translation_scale_m=float(raw["target_translation_scale_m"]),
        category_counts=tuple(int(v) for v in raw["category_counts"]),
        residual_dim=int(raw.get("residual_dim", 3)),
        stage_count=int(raw.get("stage_count", M10_STAGE_COUNT)),
        maximum_abs_residual_estimate_normalized=float(
            raw.get("maximum_abs_residual_estimate_normalized", 0.25)
        ),
        close_adapter_hidden_dim=int(raw.get("close_adapter_hidden_dim", 64)),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    residual_bound = residual.get("maximum_abs_prediction_normalized")
    if (
        not isinstance(residual_bound, (int, float))
        or not math.isfinite(float(residual_bound))
    ):
        raise RuntimeError("M11-B-D6 residual bound contract mismatch")
    if not math.isclose(
        float(residual_bound),
        config.maximum_abs_residual_estimate_normalized,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("M11-B-D6 residual bound checkpoint mismatch")
    actor = M11BoundedEventActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload


def load_m11_d8_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11BoundedEventActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D8_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D8 checkpoint schema")
    residual = payload.get("residual_estimator_contract", {})
    if residual.get("schema") != M11_D6_RESIDUAL_SCHEMA:
        raise RuntimeError("M11-B-D8 residual estimator contract mismatch")
    if residual.get("causal_recurrent_history_only") is not True:
        raise RuntimeError("M11-B-D8 residual estimator must be causal")
    if residual.get("teacher_residual_is_training_target_only") is not True:
        raise RuntimeError("M11-B-D8 teacher residual must remain training-only")
    if residual.get("runtime_accumulator_used") is not False:
        raise RuntimeError("M11-B-D8 forbids a runtime residual accumulator")
    if residual.get("predicted_intent_accumulated") is not False:
        raise RuntimeError("M11-B-D8 forbids predicted-intent accumulation")

    controller = payload.get("controller_contract", {})
    if controller.get("schema") != M11_D8_CONTROLLER_SCHEMA:
        raise RuntimeError("M11-B-D8 temporal controller contract mismatch")
    if controller.get("confirmation_mode") != "consecutive_threshold_requests":
        raise RuntimeError("M11-B-D8 CLOSE confirmation mode mismatch")
    confirmation_steps = controller.get("confirmation_steps")
    if not isinstance(confirmation_steps, int) or confirmation_steps < 2:
        raise RuntimeError("M11-B-D8 confirmation_steps contract mismatch")
    if controller.get("target_evidence_window_ends_at_expert_close") is not True:
        raise RuntimeError("M11-B-D8 training/runtime confirmation alignment mismatch")
    if controller.get("uses_future_score_or_local_peak") is not False:
        raise RuntimeError("M11-B-D8 forbids future-score/local-peak lookahead")
    if controller.get("isolated_short_runs_are_suppressed") is not True:
        raise RuntimeError("M11-B-D8 short-run suppression contract mismatch")
    if controller.get("automatic_open_on_pre_action_ready") is not True:
        raise RuntimeError("M11-B-D8 automatic causal OPEN contract mismatch")
    if controller.get("learned_open_class_used") is not False:
        raise RuntimeError("M11-B-D8 must not restore learned OPEN")
    if controller.get("post_action_truth_used_for_current_action") is not False:
        raise RuntimeError("M11-B-D8 forbids post-action truth leakage")
    if controller.get("maximum_close_emissions_per_episode") != 1:
        raise RuntimeError("M11-B-D8 CLOSE one-shot contract mismatch")
    if controller.get("maximum_open_emissions_per_episode") != 1:
        raise RuntimeError("M11-B-D8 OPEN one-shot contract mismatch")
    allowed = controller.get("allowed_close_stage_ids")
    if (
        not isinstance(allowed, (list, tuple))
        or not allowed
        or any(
            not isinstance(value, int) or value < 0 or value >= M10_STAGE_COUNT
            for value in allowed
        )
    ):
        raise RuntimeError("M11-B-D8 CLOSE-stage contract mismatch")
    threshold = controller.get("close_score_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise RuntimeError("M11-B-D8 CLOSE threshold contract mismatch")

    training = payload.get("training_contract", {})
    if training.get("translation_checkpoint_reused_without_retraining") is not True:
        raise RuntimeError("M11-B-D8 translation reuse contract mismatch")
    if training.get("event_phase_backbone_frozen") is not True:
        raise RuntimeError("M11-B-D8 frozen backbone contract mismatch")
    if training.get("close_confirmation_window_matches_runtime_counter") is not True:
        raise RuntimeError("M11-B-D8 event training/controller mismatch")

    raw = payload["model_config"]
    config = M11BoundedEventModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(int(v) for v in raw["observation_hidden_dims"]),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        target_translation_scale_m=float(raw["target_translation_scale_m"]),
        category_counts=tuple(int(v) for v in raw["category_counts"]),
        residual_dim=int(raw.get("residual_dim", 3)),
        stage_count=int(raw.get("stage_count", M10_STAGE_COUNT)),
        maximum_abs_residual_estimate_normalized=float(
            raw.get("maximum_abs_residual_estimate_normalized", 0.25)
        ),
        close_adapter_hidden_dim=int(raw.get("close_adapter_hidden_dim", 64)),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    residual_bound = residual.get("maximum_abs_prediction_normalized")
    if (
        not isinstance(residual_bound, (int, float))
        or not math.isfinite(float(residual_bound))
        or not math.isclose(
            float(residual_bound),
            config.maximum_abs_residual_estimate_normalized,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise RuntimeError("M11-B-D8 residual bound checkpoint mismatch")
    actor = M11BoundedEventActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
