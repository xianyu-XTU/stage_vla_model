"""M11-B-D5 residual-conditioned XYZ actor and causal GRIP event controller."""

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
    M9B_TRANSLATION_CENTER_TOKEN,
    M9B_TRANSLATION_MAX_ABS_BIN,
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

M11_D5_CHECKPOINT_SCHEMA = "stage_vla.m11.residual_conditioned_causal_actor.v1"
M11_D5_SUMMARY_SCHEMA = "stage_vla.m11.residual_conditioned_causal_summary.v1"
M11_D5_CONTROLLER_SCHEMA = "stage_vla.m11.causal_grip_latch.v1"
M11_D5_RESIDUAL_SCHEMA = "stage_vla.m11.bounded_execution_residual.v1"


@dataclass(frozen=True)
class M11ResidualLatchModelConfig:
    observation_dim: int = 99
    observation_hidden_dims: tuple[int, ...] = (256, 256)
    recurrent_hidden_dim: int = 256
    head_hidden_dim: int = 128
    target_translation_scale_m: float = 0.004
    category_counts: tuple[int, ...] = M9B_CATEGORY_COUNTS
    residual_dim: int = 3
    stage_count: int = M10_STAGE_COUNT
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
            raise ValueError("D5 requires one residual value for each XYZ factor")
        if self.stage_count != M10_STAGE_COUNT:
            raise ValueError("D5 requires the unchanged five-stage M10 contract")


class M11ResidualLatchActor(nn.Module):
    """D4 backbone with residual-conditioned XYZ and causal GRIP context."""

    def __init__(self, config: M11ResidualLatchModelConfig) -> None:
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
        self.translation_token_head = nn.Linear(
            config.head_hidden_dim + config.residual_dim, 15
        )
        self.intent_head = nn.Linear(config.head_hidden_dim, 3)
        self.grip_head = nn.Linear(
            config.head_hidden_dim + config.stage_count + 1, 3
        )

    def forward_sequence(
        self,
        observations: Tensor,
        residual_before_normalized: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        residual = torch.as_tensor(
            residual_before_normalized, device=value.device, dtype=value.dtype
        )
        stages = torch.as_tensor(stage_ids, device=value.device, dtype=torch.long)
        ready = torch.as_tensor(release_ready, device=value.device, dtype=torch.bool)
        if value.ndim != 3 or value.shape[-1] != self.config.observation_dim:
            raise ValueError("observations must have shape [B,T,99]")
        if residual.shape != (*value.shape[:2], self.config.residual_dim):
            raise ValueError("residual must have shape [B,T,3]")
        if stages.shape != value.shape[:2] or ready.shape != value.shape[:2]:
            raise ValueError("stage_ids/release_ready must align with [B,T]")
        if not torch.isfinite(value).all() or not torch.isfinite(residual).all():
            raise ValueError("observations/residual contain NaN/Inf")
        encoded = self.observation_encoder(value)
        recurrent, next_hidden = self.gru(encoded, hidden)
        latent = self.shared_head(recurrent)
        translation_logits = self.translation_token_head(torch.cat((latent, residual), dim=-1))
        intent = torch.tanh(self.intent_head(latent))
        stage_context = stage_one_hot(stages, dtype=latent.dtype).to(latent.device)
        grip_context = torch.cat(
            (latent, stage_context, ready.to(latent.dtype).unsqueeze(-1)), dim=-1
        )
        grip_logits = self.grip_head(grip_context)
        return translation_logits, intent, grip_logits, next_hidden

    def step(
        self,
        observations: Tensor,
        residual_before_normalized: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 2:
            raise ValueError("step expects observations [N,99]")
        xyz, intent, grip, next_hidden = self.forward_sequence(
            value.unsqueeze(1),
            torch.as_tensor(residual_before_normalized).unsqueeze(1),
            torch.as_tensor(stage_ids).unsqueeze(1),
            torch.as_tensor(release_ready).unsqueeze(1),
            hidden,
        )
        return xyz[:, 0], intent[:, 0], grip[:, 0], next_hidden


def initialize_from_d4(
    actor: M11ResidualLatchActor,
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
        raise RuntimeError("D4 checkpoint architecture does not match D5 backbone")
    actor.observation_encoder.load_state_dict(
        teacher.observation_encoder.state_dict(), strict=True
    )
    actor.gru.load_state_dict(teacher.gru.state_dict(), strict=True)
    actor.shared_head.load_state_dict(teacher.shared_head.state_dict(), strict=True)
    actor.intent_head.load_state_dict(teacher.intent_head.state_dict(), strict=True)
    with torch.no_grad():
        actor.translation_token_head.weight.zero_()
        actor.translation_token_head.weight[:, : actor.config.head_hidden_dim].copy_(
            teacher.translation_token_head.weight
        )
        actor.translation_token_head.bias.copy_(teacher.translation_token_head.bias)
        actor.grip_head.weight.zero_()
        actor.grip_head.weight[:, : actor.config.head_hidden_dim].copy_(
            teacher.grip_head.weight
        )
        actor.grip_head.bias.copy_(teacher.grip_head.bias)
    return payload


class M11BoundedExecutionResidual:
    """Causal residual updated from predicted intent and actually executed XYZ."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        maximum_abs_normalized: float = 0.5,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        if not math.isfinite(maximum_abs_normalized) or maximum_abs_normalized <= 0.25:
            raise ValueError("maximum_abs_normalized must be finite and > 0.25")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.maximum_abs_normalized = float(maximum_abs_normalized)
        self.residual = torch.zeros(self.num_envs, 3, device=self.device)
        self.active_components = 0
        self.clipped_components = 0
        self.maximum_abs_before_clamp = 0.0

    @property
    def value(self) -> Tensor:
        return self.residual.clone()

    def reset(self, mask: Tensor | None = None) -> None:
        selected = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if selected.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [num_envs]")
        self.residual[selected] = 0.0
        if mask is None:
            self.active_components = 0
            self.clipped_components = 0
            self.maximum_abs_before_clamp = 0.0

    def step(
        self,
        predicted_intent_normalized: Tensor,
        executed_translation_tokens: Tensor,
        active_mask: Tensor | None = None,
    ) -> Tensor:
        intent = torch.as_tensor(
            predicted_intent_normalized, device=self.device, dtype=self.residual.dtype
        )
        tokens = torch.as_tensor(
            executed_translation_tokens, device=self.device, dtype=torch.long
        )
        active = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if intent.shape != (self.num_envs, 3) or tokens.shape != (self.num_envs, 3):
            raise ValueError("residual update expects intent/tokens [N,3]")
        if active.shape != (self.num_envs,):
            raise ValueError("active_mask must have shape [num_envs]")
        if not torch.isfinite(intent).all():
            raise ValueError("predicted intent contains NaN/Inf")
        if bool(((tokens < 0) | (tokens > 4)).any().item()):
            raise ValueError("translation tokens must be inside [0,4]")
        executed = (
            tokens.to(self.residual.dtype) - float(M9B_TRANSLATION_CENTER_TOKEN)
        ) / float(M9B_TRANSLATION_MAX_ABS_BIN)
        candidate = self.residual + intent - executed
        if bool(active.any().item()):
            active_values = candidate[active]
            self.maximum_abs_before_clamp = max(
                self.maximum_abs_before_clamp,
                float(active_values.abs().max().item()),
            )
            clipped = active_values.abs() > self.maximum_abs_normalized
            self.active_components += int(active_values.numel())
            self.clipped_components += int(clipped.sum().item())
            self.residual[active] = active_values.clamp(
                -self.maximum_abs_normalized, self.maximum_abs_normalized
            )
        return self.value

    def summary(self) -> dict[str, Any]:
        return {
            "schema": M11_D5_RESIDUAL_SCHEMA,
            "maximum_abs_normalized": self.maximum_abs_normalized,
            "maximum_abs_before_clamp": self.maximum_abs_before_clamp,
            "active_components": self.active_components,
            "clipped_components": self.clipped_components,
            "clipped_fraction": (
                self.clipped_components / self.active_components
                if self.active_components
                else 0.0
            ),
        }


class M11CausalGripController:
    """Stage-gated CLOSE and pending OPEN latch using pre-action state only."""

    OPEN_STATE = 0
    CLOSED_STATE = 1
    RELEASED_STATE = 2

    def __init__(
        self,
        num_envs: int,
        *,
        allowed_close_stage_ids: Sequence[int],
        device: str | torch.device = "cpu",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        allowed = tuple(sorted({int(value) for value in allowed_close_stage_ids}))
        if not allowed or any(value < 0 or value >= M10_STAGE_COUNT for value in allowed):
            raise ValueError("allowed_close_stage_ids must contain valid M10 stages")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.allowed_close_stage_ids = allowed
        self.state = torch.full(
            (self.num_envs,), self.OPEN_STATE, dtype=torch.long, device=self.device
        )
        self.pending_open = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.close_emissions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.open_emissions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.pending_open_requests = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.blocked_close_requests = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(self, mask: Tensor | None = None) -> None:
        selected = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if selected.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [num_envs]")
        self.state[selected] = self.OPEN_STATE
        self.pending_open[selected] = False
        self.close_emissions[selected] = 0
        self.open_emissions[selected] = 0
        self.pending_open_requests[selected] = 0
        self.blocked_close_requests[selected] = 0

    def step(
        self,
        grip_logits: Tensor,
        stage_ids: Tensor,
        release_ready: Tensor,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = torch.as_tensor(grip_logits, device=self.device)
        stages = torch.as_tensor(stage_ids, dtype=torch.long, device=self.device).reshape(-1)
        ready = torch.as_tensor(release_ready, dtype=torch.bool, device=self.device).reshape(-1)
        active = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        expected = (self.num_envs,)
        if logits.shape != (self.num_envs, 3):
            raise ValueError("controller expects grip_logits [N,3]")
        if stages.shape != expected or ready.shape != expected or active.shape != expected:
            raise ValueError("stage/ready/active must have shape [N]")
        raw = torch.argmax(logits, dim=-1)
        close_stage = torch.zeros_like(active)
        for stage_id in self.allowed_close_stage_ids:
            close_stage |= stages == int(stage_id)
        raw_close = active & (self.state == self.OPEN_STATE) & (raw == M9B_CLOSE_TOKEN)
        blocked_close = raw_close & ~close_stage
        close_command = raw_close & close_stage
        self.blocked_close_requests += blocked_close.to(torch.long)
        self.state[close_command] = self.CLOSED_STATE

        new_open_request = (
            active
            & (self.state == self.CLOSED_STATE)
            & (raw == M9B_OPEN_TOKEN)
            & ~self.pending_open
        )
        self.pending_open[new_open_request] = True
        self.pending_open_requests += new_open_request.to(torch.long)
        open_command = (
            active
            & (self.state == self.CLOSED_STATE)
            & self.pending_open
            & ready
        )

        tokens = torch.full(
            (self.num_envs,), M9B_KEEP_TOKEN, dtype=torch.long, device=self.device
        )
        tokens[close_command] = M9B_CLOSE_TOKEN
        tokens[open_command] = M9B_OPEN_TOKEN
        self.state[open_command] = self.RELEASED_STATE
        self.pending_open[open_command] = False
        self.close_emissions += close_command.to(torch.long)
        self.open_emissions += open_command.to(torch.long)
        execution_logits = torch.full_like(logits, -20.0)
        execution_logits.scatter_(1, tokens.unsqueeze(-1), 20.0)
        return tokens, execution_logits

    def summary(self) -> dict[str, Any]:
        return {
            "schema": M11_D5_CONTROLLER_SCHEMA,
            "environments": self.num_envs,
            "allowed_close_stage_ids": list(self.allowed_close_stage_ids),
            "closed_state_environments": int((self.state >= self.CLOSED_STATE).sum().item()),
            "released_state_environments": int((self.state == self.RELEASED_STATE).sum().item()),
            "pending_open_environments": int(self.pending_open.sum().item()),
            "close_emissions": int(self.close_emissions.sum().item()),
            "open_emissions": int(self.open_emissions.sum().item()),
            "pending_open_requests": int(self.pending_open_requests.sum().item()),
            "blocked_close_requests": int(self.blocked_close_requests.sum().item()),
            "maximum_close_emissions_per_environment": int(self.close_emissions.max().item()),
            "maximum_open_emissions_per_environment": int(self.open_emissions.max().item()),
        }


@dataclass(frozen=True)
class M11CausalFilterReport:
    split: str
    original_episode_count: int
    kept_episode_count: int
    dropped_episode_ids: tuple[int, ...]
    close_target_stage_ids: tuple[int, ...]


def _episode_contract_error(tokens: Tensor, pre_ready: Tensor) -> str | None:
    close_indices = torch.nonzero(tokens[:, 3] == M9B_CLOSE_TOKEN, as_tuple=False).flatten()
    open_indices = torch.nonzero(tokens[:, 3] == M9B_OPEN_TOKEN, as_tuple=False).flatten()
    if close_indices.numel() != 1 or open_indices.numel() != 1:
        return "expected exactly one CLOSE and one OPEN"
    close_index = int(close_indices[0].item())
    open_index = int(open_indices[0].item())
    if close_index >= open_index:
        return "CLOSE must precede OPEN"
    if not bool(pre_ready[open_index].item()):
        return "expert OPEN is not pre-action release-ready"
    return None


def filter_causal_grip_split(
    split: M11IntentSplit,
) -> tuple[M11IntentSplit, M11CausalFilterReport]:
    """Drop only episodes that cannot be represented by a causal ready gate."""
    episode_ids = [int(value) for value in torch.unique(split.episode_ids, sorted=True).tolist()]
    seed_by_id = dict(zip(episode_ids, split.episode_seeds, strict=True))
    kept_ids: list[int] = []
    dropped_ids: list[int] = []
    close_stages: set[int] = set()
    lengths: list[int] = []
    for episode_id in episode_ids:
        selected = split.episode_ids == episode_id
        tokens = split.tokens[selected]
        ready = split.pre_release_ready[selected]
        error = _episode_contract_error(tokens, ready)
        if error is not None:
            dropped_ids.append(episode_id)
            continue
        kept_ids.append(episode_id)
        lengths.append(int(selected.sum().item()))
        close_index = int(
            torch.nonzero(tokens[:, 3] == M9B_CLOSE_TOKEN, as_tuple=False)[0].item()
        )
        close_stages.add(int(split.stage_ids[selected][close_index].item()))
    if not kept_ids:
        raise RuntimeError(f"{split.name} has no causally compatible GRIP episodes")
    keep_mask = torch.zeros_like(split.episode_ids, dtype=torch.bool)
    for episode_id in kept_ids:
        keep_mask |= split.episode_ids == episode_id
    filtered = M11IntentSplit(
        name=split.name,
        observations=split.observations[keep_mask],
        tokens=split.tokens[keep_mask],
        ideal_translation_normalized=split.ideal_translation_normalized[keep_mask],
        residual_before_normalized=split.residual_before_normalized[keep_mask],
        pre_release_ready=split.pre_release_ready[keep_mask],
        stage_ids=split.stage_ids[keep_mask],
        episode_step_ids=split.episode_step_ids[keep_mask],
        episode_ids=split.episode_ids[keep_mask],
        episode_seeds=tuple(seed_by_id[episode_id] for episode_id in kept_ids),
        episode_count=len(kept_ids),
        step_count=int(keep_mask.sum().item()),
        observation_dim=split.observation_dim,
        minimum_episode_steps=min(lengths),
        maximum_episode_steps=max(lengths),
    )
    report = M11CausalFilterReport(
        split=split.name,
        original_episode_count=split.episode_count,
        kept_episode_count=filtered.episode_count,
        dropped_episode_ids=tuple(dropped_ids),
        close_target_stage_ids=tuple(sorted(close_stages)),
    )
    return filtered, report


def filter_causal_grip_dataset(
    dataset: M11IntentDatasetBundle,
    *,
    minimum_train_episodes: int,
    minimum_validation_episodes: int,
) -> tuple[M11IntentDatasetBundle, Mapping[str, M11CausalFilterReport]]:
    train, train_report = filter_causal_grip_split(dataset.train)
    validation, validation_report = filter_causal_grip_split(dataset.validation)
    if train.episode_count < int(minimum_train_episodes):
        raise RuntimeError(
            f"causal filtering left {train.episode_count} train episodes; "
            f"need {minimum_train_episodes}"
        )
    if validation.episode_count < int(minimum_validation_episodes):
        raise RuntimeError(
            f"causal filtering left {validation.episode_count} validation episodes; "
            f"need {minimum_validation_episodes}"
        )
    filtered = M11IntentDatasetBundle(
        dataset_dir=dataset.dataset_dir,
        manifest_sha256=dataset.manifest_sha256,
        quantizer_method=dataset.quantizer_method,
        train=train,
        validation=validation,
    )
    return filtered, {"train": train_report, "validation": validation_report}


def residual_conditioned_loss(
    translation_logits: Tensor,
    intent_normalized: Tensor,
    grip_logits: Tensor,
    target_tokens: Tensor,
    target_intent_normalized: Tensor,
    mask: Tensor,
    episode_step_ids: Tensor,
    *,
    xyz_class_weights: Sequence[Tensor] | None,
    translation_token_weight: float,
    intent_auxiliary_weight: float,
    grip_balanced_weight: float,
    first_step_weight: float,
    early_step_weight: float,
    early_step_cutoff: int,
    grip_open_weight: float,
    grip_keep_weight: float,
    grip_close_weight: float,
    grip_focal_gamma: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid = torch.as_tensor(mask, dtype=torch.bool, device=translation_logits.device)
    target = torch.as_tensor(target_tokens, dtype=torch.long, device=translation_logits.device)
    target_intent = torch.as_tensor(
        target_intent_normalized, dtype=intent_normalized.dtype, device=intent_normalized.device
    )
    step_ids = torch.as_tensor(episode_step_ids, dtype=torch.long, device=translation_logits.device)
    if translation_logits.shape != (*valid.shape, 15):
        raise ValueError("translation logits must align as [B,T,15]")
    if grip_logits.shape != (*valid.shape, 3) or target.shape != (*valid.shape, 4):
        raise ValueError("grip/tokens/mask are not aligned")
    if intent_normalized.shape != (*valid.shape, 3) or target_intent.shape != intent_normalized.shape:
        raise ValueError("intent targets are not aligned")
    if step_ids.shape != valid.shape or not bool(valid.any().item()):
        raise ValueError("episode step ids/mask are invalid")
    if xyz_class_weights is not None and len(xyz_class_weights) != 3:
        raise ValueError("xyz_class_weights must contain three tensors")
    if early_step_cutoff < 1 or grip_focal_gamma < 0.0:
        raise ValueError("early_step_cutoff/focal gamma are invalid")
    multipliers = (float(grip_open_weight), float(grip_keep_weight), float(grip_close_weight))
    if any(not math.isfinite(value) or value <= 0.0 for value in multipliers):
        raise ValueError("all GRIP class weights must be finite and > 0")
    per_step_weight = torch.ones(valid.shape, device=valid.device)
    early = (step_ids > 0) & (step_ids < int(early_step_cutoff))
    per_step_weight[early] = float(early_step_weight)
    per_step_weight[step_ids == 0] = float(first_step_weight)
    xyz_losses: list[Tensor] = []
    for factor, part in enumerate(split_factor_logits(translation_logits, (5, 5, 5))):
        weight = None
        if xyz_class_weights is not None:
            weight = torch.as_tensor(
                xyz_class_weights[factor], dtype=part.dtype, device=part.device
            )
        loss = F.cross_entropy(
            part.reshape(-1, 5), target[..., factor].reshape(-1), weight=weight, reduction="none"
        ).reshape(valid.shape)
        xyz_losses.append(
            (loss[valid] * per_step_weight[valid]).sum() / per_step_weight[valid].sum()
        )
    xyz_loss = torch.stack(xyz_losses).mean()
    intent_loss = F.smooth_l1_loss(intent_normalized[valid], target_intent[valid])
    grip_ce = F.cross_entropy(
        grip_logits.reshape(-1, 3), target[..., 3].reshape(-1), reduction="none"
    ).reshape(valid.shape)
    grip_focal = (1.0 - torch.exp(-grip_ce)).pow(float(grip_focal_gamma)) * grip_ce
    class_losses: list[Tensor] = []
    class_denominator = 0.0
    for token, multiplier in zip(
        (M9B_OPEN_TOKEN, M9B_KEEP_TOKEN, M9B_CLOSE_TOKEN), multipliers, strict=True
    ):
        selected = valid & (target[..., 3] == token)
        if bool(selected.any().item()):
            class_losses.append(multiplier * grip_focal[selected].mean())
            class_denominator += multiplier
    grip_loss = torch.stack(class_losses).sum() / class_denominator
    total = (
        float(translation_token_weight) * xyz_loss
        + float(intent_auxiliary_weight) * intent_loss
        + float(grip_balanced_weight) * grip_loss
    )
    return total, {
        "translation_token_ce": xyz_loss,
        "intent_auxiliary": intent_loss,
        "grip_balanced_focal": grip_loss,
    }


@dataclass(frozen=True)
class M11ResidualLatchEvaluation:
    autonomous_metrics: M11HybridMetrics
    raw_autonomous_metrics: M11HybridMetrics
    teacher_conditioned_metrics: M11HybridMetrics
    resolved_grip_state_accuracy: float
    controller_summary: Mapping[str, Any]
    residual_summary: Mapping[str, Any]
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


def _first_event_index(tokens: Tensor, event: int) -> int | None:
    indices = torch.nonzero(tokens[:, 3] == int(event), as_tuple=False).flatten()
    return int(indices[0].item()) if indices.numel() else None


def evaluate_residual_latch_split(
    actor: M11ResidualLatchActor,
    split: M11IntentSplit,
    device: str | torch.device,
    *,
    allowed_close_stage_ids: Sequence[int],
    maximum_abs_residual_normalized: float,
) -> M11ResidualLatchEvaluation:
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
        raise RuntimeError("D5 evaluation split contains no episodes")
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
    target_residual = torch.zeros(num_episodes, max_steps, 3, device=target_device)
    target_tokens = torch.zeros(num_episodes, max_steps, 4, dtype=torch.long)
    stages = torch.zeros(num_episodes, max_steps, dtype=torch.long, device=target_device)
    ready = torch.zeros(num_episodes, max_steps, dtype=torch.bool, device=target_device)
    for index, (_id, obs, intent, residual, tokens, stage, pre_ready) in enumerate(episodes):
        length = int(tokens.shape[0])
        observations[index, :length] = obs.to(target_device)
        target_intent[index, :length] = intent
        target_residual[index, :length] = residual.to(target_device)
        target_tokens[index, :length] = tokens
        stages[index, :length] = stage.to(target_device)
        ready[index, :length] = pre_ready.to(target_device)

    actor.eval()
    with torch.inference_mode():
        teacher_xyz, teacher_intent, teacher_grip, _ = actor.forward_sequence(
            observations, target_residual, stages, ready
        )
        teacher_tokens = torch.cat(
            (
                translation_tokens(teacher_xyz),
                torch.argmax(teacher_grip, dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        controller = M11CausalGripController(
            num_episodes,
            allowed_close_stage_ids=allowed_close_stage_ids,
            device=target_device,
        )
        residual_state = M11BoundedExecutionResidual(
            num_episodes,
            device=target_device,
            maximum_abs_normalized=maximum_abs_residual_normalized,
        )
        hidden = None
        autonomous_tokens: list[Tensor] = []
        raw_tokens: list[Tensor] = []
        autonomous_intents: list[Tensor] = []
        for step in range(max_steps):
            active = valid[:, step]
            xyz_logits, intent, grip_logits, hidden = actor.step(
                observations[:, step],
                residual_state.value,
                stages[:, step],
                ready[:, step],
                hidden,
            )
            xyz = translation_tokens(xyz_logits)
            grip, _grip_execution_logits = controller.step(
                grip_logits,
                stages[:, step],
                ready[:, step],
                active_mask=active,
            )
            residual_state.step(intent, xyz, active_mask=active)
            autonomous_tokens.append(torch.cat((xyz, grip.unsqueeze(-1)), dim=-1))
            raw_tokens.append(
                torch.cat((xyz, torch.argmax(grip_logits, dim=-1, keepdim=True)), dim=-1)
            )
            autonomous_intents.append(intent)
        autonomous = torch.stack(autonomous_tokens, dim=1).cpu()
        raw_autonomous = torch.stack(raw_tokens, dim=1).cpu()
        autonomous_intent = torch.stack(autonomous_intents, dim=1).cpu()

    valid_cpu = valid.cpu()
    target_valid = target_tokens[valid_cpu]
    intent_target_valid = target_intent[valid_cpu]
    controller_summary = controller.summary()
    autonomous_metrics = hybrid_metrics(
        autonomous[valid_cpu],
        target_valid,
        autonomous_intent[valid_cpu],
        intent_target_valid,
        target_translation_scale_m=actor.config.target_translation_scale_m,
        close_emissions=int(controller.close_emissions.sum().item()),
        open_emissions=int(controller.open_emissions.sum().item()),
    )
    raw_metrics = hybrid_metrics(
        raw_autonomous[valid_cpu],
        target_valid,
        autonomous_intent[valid_cpu],
        intent_target_valid,
        target_translation_scale_m=actor.config.target_translation_scale_m,
        close_emissions=int(
            (raw_autonomous[valid_cpu][:, 3] == M9B_CLOSE_TOKEN).sum().item()
        ),
        open_emissions=int(
            (raw_autonomous[valid_cpu][:, 3] == M9B_OPEN_TOKEN).sum().item()
        ),
    )
    teacher_cpu = teacher_tokens.cpu()
    teacher_metrics = hybrid_metrics(
        teacher_cpu[valid_cpu],
        target_valid,
        teacher_intent.cpu()[valid_cpu],
        intent_target_valid,
        target_translation_scale_m=actor.config.target_translation_scale_m,
        close_emissions=int(
            (teacher_cpu[valid_cpu][:, 3] == M9B_CLOSE_TOKEN).sum().item()
        ),
        open_emissions=int(
            (teacher_cpu[valid_cpu][:, 3] == M9B_OPEN_TOKEN).sum().item()
        ),
    )

    predicted_by_episode: list[Tensor] = []
    targets_by_episode: list[Tensor] = []
    timing: list[Mapping[str, Any]] = []
    correct_states = 0
    state_steps = 0
    for index, (episode_id, _obs, _intent, _residual, tokens, _stage, _ready) in enumerate(episodes):
        length = int(tokens.shape[0])
        predicted = autonomous[index, :length].clone()
        predicted_by_episode.append(predicted)
        targets_by_episode.append(tokens.clone())
        predicted_state = _resolved_grip_state(predicted)
        target_state = _resolved_grip_state(tokens)
        correct_states += int((predicted_state == target_state).sum().item())
        state_steps += length
        target_close = _first_event_index(tokens, M9B_CLOSE_TOKEN)
        predicted_close = _first_event_index(predicted, M9B_CLOSE_TOKEN)
        target_open = _first_event_index(tokens, M9B_OPEN_TOKEN)
        predicted_open = _first_event_index(predicted, M9B_OPEN_TOKEN)
        timing.append(
            {
                "episode_id": episode_id,
                "target_close_step": target_close,
                "predicted_close_step": predicted_close,
                "close_step_error": (
                    predicted_close - target_close
                    if predicted_close is not None and target_close is not None
                    else None
                ),
                "target_open_step": target_open,
                "predicted_open_step": predicted_open,
                "open_step_error": (
                    predicted_open - target_open
                    if predicted_open is not None and target_open is not None
                    else None
                ),
            }
        )
    return M11ResidualLatchEvaluation(
        autonomous_metrics=autonomous_metrics,
        raw_autonomous_metrics=raw_metrics,
        teacher_conditioned_metrics=teacher_metrics,
        resolved_grip_state_accuracy=correct_states / state_steps if state_steps else 0.0,
        controller_summary=controller_summary,
        residual_summary=residual_state.summary(),
        release_ready_contract=grip_release_ready_contract(split),
        event_timing=tuple(timing),
        episode_ids=tuple(item[0] for item in episodes),
        predicted_tokens_by_episode=tuple(predicted_by_episode),
        target_tokens_by_episode=tuple(targets_by_episode),
    )


def d5_selection_score(evaluation: M11ResidualLatchEvaluation) -> float:
    metrics = evaluation.autonomous_metrics
    raw = evaluation.raw_autonomous_metrics
    return float(
        2.0 * (1.0 - metrics.exact_action_accuracy)
        + sum(1.0 - value for value in metrics.factor_accuracy[:3])
        + (1.0 - metrics.grip_open_recall)
        + (1.0 - metrics.grip_close_recall)
        + 0.5 * (1.0 - raw.grip_open_recall)
        + 0.5 * (1.0 - raw.grip_close_recall)
        + (1.0 - evaluation.resolved_grip_state_accuracy)
        + 0.1 * sum(metrics.translation_intent_mae_mm) / 3.0
    )


def d5_checkpoint_payload(
    actor: M11ResidualLatchActor,
    *,
    epoch: int,
    evaluation: M11ResidualLatchEvaluation,
    selection_score: float,
    dataset: M11IntentDatasetBundle,
    filter_reports: Mapping[str, M11CausalFilterReport],
    training_config: Mapping[str, Any],
    d4_checkpoint: str,
    allowed_close_stage_ids: Sequence[int],
    maximum_abs_residual_normalized: float,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": M11_D5_CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "validation_metrics": asdict(evaluation.autonomous_metrics),
        "validation_raw_metrics": asdict(evaluation.raw_autonomous_metrics),
        "validation_teacher_conditioned_metrics": asdict(
            evaluation.teacher_conditioned_metrics
        ),
        "validation_resolved_grip_state_accuracy": float(
            evaluation.resolved_grip_state_accuracy
        ),
        "validation_controller_summary": dict(evaluation.controller_summary),
        "validation_residual_summary": dict(evaluation.residual_summary),
        "selection_score": float(selection_score),
        "training_config": dict(training_config),
        "initialized_from_d4_checkpoint": str(d4_checkpoint),
        "dataset": {
            "directory": dataset.dataset_dir,
            "manifest_sha256": dataset.manifest_sha256,
            "quantizer_method": dataset.quantizer_method,
            "train_episodes": dataset.train.episode_count,
            "validation_episodes": dataset.validation.episode_count,
            "minimum_episode_steps": min(
                dataset.train.minimum_episode_steps, dataset.validation.minimum_episode_steps
            ),
            "maximum_episode_steps": max(
                dataset.train.maximum_episode_steps, dataset.validation.maximum_episode_steps
            ),
            "causal_filter": {
                name: asdict(report) for name, report in filter_reports.items()
            },
        },
        "controller_contract": {
            "schema": M11_D5_CONTROLLER_SCHEMA,
            "initial_state": "OPEN",
            "allowed_close_stage_ids": [int(value) for value in allowed_close_stage_ids],
            "pending_open_latch": True,
            "open_requires_pre_action_release_ready": True,
            "post_action_truth_used_for_current_action": False,
            "maximum_close_emissions_per_episode": 1,
            "maximum_open_emissions_per_episode": 1,
        },
        "residual_contract": {
            "schema": M11_D5_RESIDUAL_SCHEMA,
            "initial_residual_normalized": [0.0, 0.0, 0.0],
            "updated_from_predicted_intent_and_executed_xyz": True,
            "maximum_abs_normalized": float(maximum_abs_residual_normalized),
        },
        "execution_contract": {
            "xyz_head_receives_causal_residual": True,
            "grip_head_receives_stage_and_pre_action_ready": True,
            "previous_token_input": False,
            "ppo_authorized": False,
        },
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    return payload


def load_m11_d5_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11ResidualLatchActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D5_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D5 checkpoint schema")
    controller = payload.get("controller_contract", {})
    if controller.get("schema") != M11_D5_CONTROLLER_SCHEMA:
        raise RuntimeError("M11-B-D5 controller contract mismatch")
    if controller.get("pending_open_latch") is not True:
        raise RuntimeError("M11-B-D5 pending OPEN contract mismatch")
    if controller.get("open_requires_pre_action_release_ready") is not True:
        raise RuntimeError("M11-B-D5 pre-action ready contract mismatch")
    if controller.get("post_action_truth_used_for_current_action") is not False:
        raise RuntimeError("M11-B-D5 forbids post-action truth leakage")
    allowed_close_stage_ids = controller.get("allowed_close_stage_ids")
    if (
        not isinstance(allowed_close_stage_ids, (list, tuple))
        or not allowed_close_stage_ids
        or any(
            not isinstance(value, int) or value < 0 or value >= M10_STAGE_COUNT
            for value in allowed_close_stage_ids
        )
    ):
        raise RuntimeError("M11-B-D5 CLOSE-stage contract mismatch")
    residual = payload.get("residual_contract", {})
    if residual.get("schema") != M11_D5_RESIDUAL_SCHEMA:
        raise RuntimeError("M11-B-D5 residual contract mismatch")
    if residual.get("initial_residual_normalized") != [0.0, 0.0, 0.0]:
        raise RuntimeError("M11-B-D5 initial residual contract mismatch")
    if residual.get("updated_from_predicted_intent_and_executed_xyz") is not True:
        raise RuntimeError("M11-B-D5 residual update contract mismatch")
    maximum_abs_residual = residual.get("maximum_abs_normalized")
    if (
        not isinstance(maximum_abs_residual, (int, float))
        or not math.isfinite(float(maximum_abs_residual))
        or float(maximum_abs_residual) <= 0.25
    ):
        raise RuntimeError("M11-B-D5 residual bound contract mismatch")
    raw = payload["model_config"]
    config = M11ResidualLatchModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(int(v) for v in raw["observation_hidden_dims"]),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        target_translation_scale_m=float(raw["target_translation_scale_m"]),
        category_counts=tuple(int(v) for v in raw["category_counts"]),
        residual_dim=int(raw.get("residual_dim", 3)),
        stage_count=int(raw.get("stage_count", M10_STAGE_COUNT)),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    actor = M11ResidualLatchActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
