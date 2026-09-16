"""M11-B-D2 history-conditioned factorized behavior-cloning actor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from stage_vla.rl.action_dsl import (
    M9B_CATEGORY_COUNTS,
    M9B_OPEN_TOKEN,
    M9B_TRANSLATION_CENTER_TOKEN,
)
from stage_vla.rl.factorized_categorical_core import (
    deterministic_factor_tokens,
    split_factor_logits,
    validate_factor_tokens,
)

from .m11_behavior_cloning import M11BCDatasetBundle, M11BCMetrics, metrics_from_logits
from .m11_divergence_audit import TokenDivergence, token_divergence

M11_D2_CHECKPOINT_SCHEMA = "stage_vla.m11.history_factorized_bc_actor.v1"
M11_D2_SUMMARY_SCHEMA = "stage_vla.m11.history_factorized_bc_summary.v1"
M11_D2_INITIAL_PREVIOUS_TOKEN = (
    M9B_TRANSLATION_CENTER_TOKEN,
    M9B_TRANSLATION_CENTER_TOKEN,
    M9B_TRANSLATION_CENTER_TOKEN,
    M9B_OPEN_TOKEN,
)


@dataclass(frozen=True)
class M11HistoryModelConfig:
    observation_dim: int = 99
    observation_hidden_dims: tuple[int, ...] = (256, 128)
    token_embedding_dim: int = 8
    recurrent_hidden_dim: int = 128
    head_hidden_dim: int = 64
    category_counts: tuple[int, ...] = M9B_CATEGORY_COUNTS
    activation: str = "elu"
    observation_normalization: bool = False

    def __post_init__(self) -> None:
        if self.observation_dim != 99:
            raise ValueError("M11-B-D2 requires the exact 99-D F1 observation")
        if tuple(int(value) for value in self.category_counts) != tuple(M9B_CATEGORY_COUNTS):
            raise ValueError("M11-B-D2 must preserve the [5,5,5,3] Action DSL")
        positive = (
            *self.observation_hidden_dims,
            self.token_embedding_dim,
            self.recurrent_hidden_dim,
            self.head_hidden_dim,
        )
        if not self.observation_hidden_dims or any(int(value) <= 0 for value in positive):
            raise ValueError("all hidden/embedding dimensions must be positive")
        if self.activation.lower() != "elu":
            raise ValueError("M11-B-D2 uses ELU to preserve the F1 nonlinear family")
        if self.observation_normalization:
            raise ValueError("M11-B-D2 keeps F1 observation_normalization=False")


def initial_previous_tokens(
    batch_size: int,
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    return torch.tensor(
        M11_D2_INITIAL_PREVIOUS_TOKEN,
        dtype=torch.long,
        device=device,
    ).reshape(1, 4).repeat(int(batch_size), 1)


def shifted_previous_tokens(tokens: Tensor) -> Tensor:
    value = validate_factor_tokens(tokens, M9B_CATEGORY_COUNTS)
    if value.ndim == 2:
        # Here the leading dimension represents batch, so this helper requires
        # explicit [B,T,4] input to avoid ambiguous trajectory semantics.
        raise ValueError("shifted_previous_tokens requires [B,T,4] tokens")
    if value.ndim != 3:
        raise ValueError("tokens must have shape [B,T,4]")
    start = initial_previous_tokens(value.shape[0], device=value.device).unsqueeze(1)
    return torch.cat((start, value[:, :-1]), dim=1)


class M11HistoryFactorizedActor(nn.Module):
    """Observation encoder + previous-token embeddings + one-layer GRU."""

    def __init__(self, config: M11HistoryModelConfig) -> None:
        super().__init__()
        self.config = config
        widths = (config.observation_dim, *config.observation_hidden_dims)
        encoder: list[nn.Module] = []
        for index in range(len(widths) - 1):
            encoder.extend((nn.Linear(widths[index], widths[index + 1]), nn.ELU()))
        self.observation_encoder = nn.Sequential(*encoder)
        self.token_embeddings = nn.ModuleList(
            nn.Embedding(int(count), config.token_embedding_dim)
            for count in config.category_counts
        )
        recurrent_input_dim = widths[-1] + len(config.category_counts) * config.token_embedding_dim
        self.gru = nn.GRU(
            input_size=recurrent_input_dim,
            hidden_size=config.recurrent_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.recurrent_hidden_dim, config.head_hidden_dim),
            nn.ELU(),
            nn.Linear(config.head_hidden_dim, sum(config.category_counts)),
        )

    def _encode(self, observations: Tensor, previous_tokens: Tensor) -> Tensor:
        obs = torch.as_tensor(observations)
        tokens = validate_factor_tokens(previous_tokens, self.config.category_counts).to(obs.device)
        if obs.ndim != 3 or obs.shape[-1] != self.config.observation_dim:
            raise ValueError("observations must have shape [B,T,99]")
        if tokens.shape != (*obs.shape[:2], len(self.config.category_counts)):
            raise ValueError("previous_tokens must align as [B,T,4]")
        if not torch.isfinite(obs).all():
            raise ValueError("observations contain NaN/Inf")
        encoded = self.observation_encoder(obs)
        embedded = torch.cat(
            [embedding(tokens[..., index]) for index, embedding in enumerate(self.token_embeddings)],
            dim=-1,
        )
        return torch.cat((encoded, embedded), dim=-1)

    def forward_sequence(
        self,
        observations: Tensor,
        previous_tokens: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        recurrent_input = self._encode(observations, previous_tokens)
        recurrent, next_hidden = self.gru(recurrent_input, hidden)
        return self.head(recurrent), next_hidden

    def step(
        self,
        observations: Tensor,
        previous_tokens: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        obs = torch.as_tensor(observations)
        prev = torch.as_tensor(previous_tokens)
        if obs.ndim != 2 or prev.ndim != 2:
            raise ValueError("step expects observations [N,99] and previous_tokens [N,4]")
        logits, next_hidden = self.forward_sequence(
            obs.unsqueeze(1), prev.unsqueeze(1), hidden
        )
        return logits[:, 0], next_hidden

    def deterministic_step(
        self,
        observations: Tensor,
        previous_tokens: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, next_hidden = self.step(observations, previous_tokens, hidden)
        tokens = deterministic_factor_tokens(logits, self.config.category_counts)
        return tokens, logits, next_hidden


def corrupt_previous_tokens(
    previous_tokens: Tensor,
    *,
    probability: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be inside [0,1]")
    value = validate_factor_tokens(previous_tokens, M9B_CATEGORY_COUNTS).clone()
    if probability == 0.0:
        return value
    random = torch.rand(value.shape, device=value.device, generator=generator)
    for factor, count in enumerate(M9B_CATEGORY_COUNTS):
        replacement = torch.randint(
            0,
            int(count),
            value[..., factor].shape,
            device=value.device,
            generator=generator,
        )
        value[..., factor] = torch.where(
            random[..., factor] < probability,
            replacement,
            value[..., factor],
        )
    return value


def masked_factorized_ce_loss(
    logits: Tensor,
    tokens: Tensor,
    mask: Tensor,
    *,
    class_weights: Sequence[Tensor] | None = None,
) -> Tensor:
    if logits.ndim != 3 or logits.shape[-1] != sum(M9B_CATEGORY_COUNTS):
        raise ValueError("logits must have shape [B,T,18]")
    target = validate_factor_tokens(tokens, M9B_CATEGORY_COUNTS).to(logits.device)
    valid = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if target.shape != (*logits.shape[:2], 4) or valid.shape != logits.shape[:2]:
        raise ValueError("tokens/mask must align with logits [B,T]")
    if not bool(valid.any().item()):
        raise ValueError("loss mask contains no valid target")
    if class_weights is not None and len(class_weights) != 4:
        raise ValueError("class_weights must contain four factor tensors")
    losses: list[Tensor] = []
    for factor, part in enumerate(split_factor_logits(logits, M9B_CATEGORY_COUNTS)):
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(class_weights[factor], dtype=part.dtype, device=part.device)
        per_step = F.cross_entropy(
            part.reshape(-1, part.shape[-1]),
            target[..., factor].reshape(-1),
            weight=weight,
            reduction="none",
        ).reshape(valid.shape)
        losses.append(per_step[valid].mean())
    return torch.stack(losses).mean()


def masked_sequence_metrics(
    logits: Tensor,
    tokens: Tensor,
    mask: Tensor,
) -> tuple[M11BCMetrics, TokenDivergence]:
    valid = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    target = validate_factor_tokens(tokens, M9B_CATEGORY_COUNTS).to(logits.device)
    selected_logits = logits[valid]
    selected_tokens = target[valid]
    predictions = deterministic_factor_tokens(selected_logits, M9B_CATEGORY_COUNTS)
    return (
        metrics_from_logits(selected_logits, selected_tokens),
        token_divergence(predictions, selected_tokens),
    )


def d2_selection_score(
    teacher_metrics: M11BCMetrics,
    *,
    autoregressive_exact_action_accuracy: float | None = None,
) -> float:
    score = teacher_metrics.joint_nll + 2.0 * (1.0 - teacher_metrics.exact_action_accuracy)
    score += 0.5 * (2.0 - teacher_metrics.grip_open_recall - teacher_metrics.grip_close_recall)
    if autoregressive_exact_action_accuracy is not None:
        score += 1.0 - float(autoregressive_exact_action_accuracy)
    return float(score)


def d2_checkpoint_payload(
    actor: M11HistoryFactorizedActor,
    *,
    epoch: int,
    teacher_metrics: M11BCMetrics,
    selection_score: float,
    dataset: M11BCDatasetBundle,
    training_config: Mapping[str, Any],
    class_weights: Sequence[Tensor],
    optimizer_state_dict: Mapping[str, Any] | None = None,
    best_epoch: int | None = None,
    best_score: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": M11_D2_CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "teacher_forced_validation_metrics": asdict(teacher_metrics),
        "selection_score": float(selection_score),
        "best_epoch": int(best_epoch if best_epoch is not None else epoch),
        "best_score": float(best_score if best_score is not None else selection_score),
        "training_config": dict(training_config),
        "class_weights": [torch.as_tensor(value).cpu() for value in class_weights],
        "dataset": {
            "directory": dataset.dataset_dir,
            "manifest_sha256": dataset.manifest_sha256,
            "quantizer_method": dataset.quantizer_method,
            "train_episodes": dataset.train.episode_count,
            "validation_episodes": dataset.validation.episode_count,
            "train_steps": dataset.train.step_count,
            "validation_steps": dataset.validation.step_count,
        },
        "runtime_contract": {
            "initial_previous_token": list(M11_D2_INITIAL_PREVIOUS_TOKEN),
            "hidden_reset_on_episode_reset": True,
            "deterministic_factorwise_argmax": True,
        },
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    return payload


def load_m11_d2_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11HistoryFactorizedActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D2_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D2 checkpoint schema")
    runtime_contract = payload.get("runtime_contract", {})
    if tuple(runtime_contract.get("initial_previous_token", ())) != tuple(
        M11_D2_INITIAL_PREVIOUS_TOKEN
    ):
        raise RuntimeError("M11-B-D2 checkpoint initial previous-token contract mismatch")
    if runtime_contract.get("hidden_reset_on_episode_reset") is not True:
        raise RuntimeError("M11-B-D2 checkpoint hidden-reset contract mismatch")
    raw = payload["model_config"]
    config = M11HistoryModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(int(value) for value in raw["observation_hidden_dims"]),
        token_embedding_dim=int(raw["token_embedding_dim"]),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        category_counts=tuple(int(value) for value in raw["category_counts"]),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    actor = M11HistoryFactorizedActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
