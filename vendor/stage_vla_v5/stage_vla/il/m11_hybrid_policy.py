"""M11-B-D4 direct token policy with continuous-intent auxiliary supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from .m11_intent_codec import (
    M11IntentDatasetBundle,
    M11IntentModelConfig,
    M11IntentSplit,
    load_m11_d3_actor,
)

M11_D4_CHECKPOINT_SCHEMA = "stage_vla.m11.hybrid_direct_token_actor.v1"
M11_D4_SUMMARY_SCHEMA = "stage_vla.m11.hybrid_direct_token_summary.v1"
M11_D4_GRIP_FSM_SCHEMA = "stage_vla.m11.one_shot_grip_fsm.v1"


@dataclass(frozen=True)
class M11HybridModelConfig:
    observation_dim: int = 99
    observation_hidden_dims: tuple[int, ...] = (256, 256)
    recurrent_hidden_dim: int = 256
    head_hidden_dim: int = 128
    target_translation_scale_m: float = 0.004
    category_counts: tuple[int, ...] = M9B_CATEGORY_COUNTS
    activation: str = "elu"
    observation_normalization: bool = False

    def __post_init__(self) -> None:
        # Reuse D3's frozen observation/scale contract validation.
        M11IntentModelConfig(
            observation_dim=self.observation_dim,
            observation_hidden_dims=self.observation_hidden_dims,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            target_translation_scale_m=self.target_translation_scale_m,
            category_counts=self.category_counts,
            activation=self.activation,
            observation_normalization=self.observation_normalization,
        )


class M11HybridTokenActor(nn.Module):
    """D3 backbone, direct XYZ categories, intent auxiliary and GRIP logits."""

    def __init__(self, config: M11HybridModelConfig) -> None:
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
        self.translation_token_head = nn.Linear(config.head_hidden_dim, 15)
        self.intent_head = nn.Linear(config.head_hidden_dim, 3)
        self.grip_head = nn.Linear(config.head_hidden_dim, 3)

    def forward_sequence(
        self,
        observations: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 3 or value.shape[-1] != self.config.observation_dim:
            raise ValueError("observations must have shape [B,T,99]")
        if not torch.isfinite(value).all():
            raise ValueError("observations contain NaN/Inf")
        encoded = self.observation_encoder(value)
        recurrent, next_hidden = self.gru(encoded, hidden)
        latent = self.shared_head(recurrent)
        translation_logits = self.translation_token_head(latent)
        intent = torch.tanh(self.intent_head(latent))
        grip_logits = self.grip_head(latent)
        return translation_logits, intent, grip_logits, next_hidden

    def step(
        self,
        observations: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 2:
            raise ValueError("step expects observations [N,99]")
        xyz, intent, grip, next_hidden = self.forward_sequence(value.unsqueeze(1), hidden)
        return xyz[:, 0], intent[:, 0], grip[:, 0], next_hidden


def initialize_from_d3(
    actor: M11HybridTokenActor,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    teacher, payload = load_m11_d3_actor(checkpoint, map_location=map_location)
    if asdict(teacher.config) != asdict(M11IntentModelConfig(
        observation_dim=actor.config.observation_dim,
        observation_hidden_dims=actor.config.observation_hidden_dims,
        recurrent_hidden_dim=actor.config.recurrent_hidden_dim,
        head_hidden_dim=actor.config.head_hidden_dim,
        target_translation_scale_m=actor.config.target_translation_scale_m,
        category_counts=actor.config.category_counts,
        activation=actor.config.activation,
        observation_normalization=actor.config.observation_normalization,
    )):
        raise RuntimeError("D3 checkpoint architecture does not match D4 backbone")
    actor.observation_encoder.load_state_dict(teacher.observation_encoder.state_dict(), strict=True)
    actor.gru.load_state_dict(teacher.gru.state_dict(), strict=True)
    actor.shared_head.load_state_dict(teacher.shared_head.state_dict(), strict=True)
    actor.intent_head.load_state_dict(teacher.translation_head.state_dict(), strict=True)
    actor.grip_head.load_state_dict(teacher.grip_head.state_dict(), strict=True)
    # Seed each direct five-way head from the corresponding D3 intent row.
    # Scores proportional to -(y-c)^2 reduce to 2*c*y-c^2, where y is the
    # pretrained linear intent projection and c is an Action-DSL bin center.
    centers = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        dtype=teacher.translation_head.weight.dtype,
        device=teacher.translation_head.weight.device,
    )
    with torch.no_grad():
        for factor in range(3):
            target = slice(5 * factor, 5 * (factor + 1))
            source_weight = teacher.translation_head.weight[factor]
            source_bias = teacher.translation_head.bias[factor]
            actor.translation_token_head.weight[target].copy_(
                2.0 * centers.unsqueeze(-1) * source_weight.unsqueeze(0)
            )
            actor.translation_token_head.bias[target].copy_(
                2.0 * centers * source_bias - centers.square()
            )
    return payload


def translation_tokens(translation_logits: Tensor) -> Tensor:
    parts = split_factor_logits(translation_logits, (5, 5, 5))
    return torch.stack([torch.argmax(part, dim=-1) for part in parts], dim=-1)


class M11OneShotGripFSM:
    """Enforce one CLOSE, persistent closed KEEP, then one ready-gated OPEN."""

    OPEN_STATE = 0
    CLOSED_STATE = 1
    RELEASED_STATE = 2

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        require_release_ready_for_open: bool = True,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.require_release_ready_for_open = bool(require_release_ready_for_open)
        self.state = torch.full(
            (self.num_envs,), self.OPEN_STATE, dtype=torch.long, device=self.device
        )
        self.close_emissions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.open_emissions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

    def step(
        self,
        grip_logits: Tensor,
        release_ready: Tensor,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = torch.as_tensor(grip_logits, device=self.device)
        ready = torch.as_tensor(release_ready, dtype=torch.bool, device=self.device).reshape(-1)
        if logits.shape != (self.num_envs, 3) or ready.shape != (self.num_envs,):
            raise ValueError("FSM expects grip_logits [N,3] and release_ready [N]")
        active = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if active.shape != (self.num_envs,):
            raise ValueError("active_mask must have shape [num_envs]")
        raw = torch.argmax(logits, dim=-1)
        close = active & (self.state == self.OPEN_STATE) & (raw == M9B_CLOSE_TOKEN)
        open_allowed = ready if self.require_release_ready_for_open else torch.ones_like(ready)
        open_command = (
            active
            & (self.state == self.CLOSED_STATE)
            & (raw == M9B_OPEN_TOKEN)
            & open_allowed
        )
        tokens = torch.full(
            (self.num_envs,), M9B_KEEP_TOKEN, dtype=torch.long, device=self.device
        )
        tokens[close] = M9B_CLOSE_TOKEN
        tokens[open_command] = M9B_OPEN_TOKEN
        self.state[close] = self.CLOSED_STATE
        self.state[open_command] = self.RELEASED_STATE
        self.close_emissions += close.to(torch.long)
        self.open_emissions += open_command.to(torch.long)
        execution_logits = torch.full_like(logits, -20.0)
        execution_logits.scatter_(1, tokens.unsqueeze(-1), 20.0)
        return tokens, execution_logits

    def summary(self) -> dict[str, int]:
        return {
            "environments": self.num_envs,
            "closed_state_environments": int((self.state >= self.CLOSED_STATE).sum().item()),
            "released_state_environments": int((self.state == self.RELEASED_STATE).sum().item()),
            "close_emissions": int(self.close_emissions.sum().item()),
            "open_emissions": int(self.open_emissions.sum().item()),
            "maximum_close_emissions_per_environment": int(self.close_emissions.max().item()),
            "maximum_open_emissions_per_environment": int(self.open_emissions.max().item()),
        }


def execution_logits(
    translation_logits: Tensor,
    grip_execution_logits: Tensor,
) -> Tensor:
    xyz_tokens = translation_tokens(translation_logits)
    parts: list[Tensor] = []
    for factor in range(3):
        part = torch.full(
            (*xyz_tokens.shape[:-1], 5),
            -20.0,
            dtype=translation_logits.dtype,
            device=translation_logits.device,
        )
        part.scatter_(-1, xyz_tokens[..., factor : factor + 1], 20.0)
        parts.append(part)
    return torch.cat((*parts, grip_execution_logits), dim=-1)


def hybrid_loss(
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
    keep_class_weight: float,
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
    per_step_weight = torch.where(
        step_ids == 0,
        torch.full(valid.shape, float(first_step_weight), device=valid.device),
        torch.ones(valid.shape, device=valid.device),
    )
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
    grip_per_step = F.cross_entropy(
        grip_logits.reshape(-1, 3), target[..., 3].reshape(-1), reduction="none"
    ).reshape(valid.shape)
    class_losses: list[Tensor] = []
    for token in (M9B_OPEN_TOKEN, M9B_KEEP_TOKEN, M9B_CLOSE_TOKEN):
        selected = valid & (target[..., 3] == token)
        if bool(selected.any().item()):
            multiplier = float(keep_class_weight) if token == M9B_KEEP_TOKEN else 1.0
            class_losses.append(multiplier * grip_per_step[selected].mean())
    grip_loss = torch.stack(class_losses).sum() / sum(
        float(keep_class_weight) if token == M9B_KEEP_TOKEN else 1.0
        for token in (M9B_OPEN_TOKEN, M9B_KEEP_TOKEN, M9B_CLOSE_TOKEN)
        if bool((valid & (target[..., 3] == token)).any().item())
    )
    total = (
        float(translation_token_weight) * xyz_loss
        + float(intent_auxiliary_weight) * intent_loss
        + float(grip_balanced_weight) * grip_loss
    )
    return total, {
        "translation_token_ce": xyz_loss,
        "intent_auxiliary": intent_loss,
        "grip_balanced_ce": grip_loss,
    }


@dataclass(frozen=True)
class M11HybridMetrics:
    num_steps: int
    factor_accuracy: tuple[float, float, float, float]
    exact_action_accuracy: float
    grip_open_recall: float
    grip_close_recall: float
    translation_intent_mae_mm: tuple[float, float, float]
    close_emissions: int
    open_emissions: int


@dataclass(frozen=True)
class M11HybridEvaluation:
    metrics: M11HybridMetrics
    raw_grip_metrics: M11HybridMetrics
    fsm_summary: Mapping[str, int]
    release_ready_contract: Mapping[str, Any]
    episode_ids: tuple[int, ...]
    predicted_tokens_by_episode: tuple[Tensor, ...]
    target_tokens_by_episode: tuple[Tensor, ...]


def hybrid_metrics(
    predicted_tokens: Tensor,
    target_tokens: Tensor,
    predicted_intent: Tensor,
    target_intent: Tensor,
    *,
    target_translation_scale_m: float,
    close_emissions: int,
    open_emissions: int,
) -> M11HybridMetrics:
    predicted = torch.as_tensor(predicted_tokens, dtype=torch.long).cpu()
    target = torch.as_tensor(target_tokens, dtype=torch.long).cpu()
    predicted_continuous = torch.as_tensor(predicted_intent).cpu()
    target_continuous = torch.as_tensor(target_intent).cpu()
    if predicted.shape != target.shape or predicted_continuous.shape != target_continuous.shape:
        raise ValueError("hybrid metric predictions and targets must align")
    factors = tuple(
        float((predicted[:, factor] == target[:, factor]).to(torch.float64).mean().item())
        for factor in range(4)
    )
    exact = float((predicted == target).all(dim=-1).to(torch.float64).mean().item())
    grip_target = target[:, 3]
    grip_predicted = predicted[:, 3]

    def recall(token: int) -> float:
        selected = grip_target == token
        return (
            float((grip_predicted[selected] == token).to(torch.float64).mean().item())
            if bool(selected.any().item())
            else 0.0
        )

    error_mm = (
        predicted_continuous - target_continuous
    ).abs() * float(target_translation_scale_m) * 1000.0
    return M11HybridMetrics(
        num_steps=int(target.shape[0]),
        factor_accuracy=factors,
        exact_action_accuracy=exact,
        grip_open_recall=recall(M9B_OPEN_TOKEN),
        grip_close_recall=recall(M9B_CLOSE_TOKEN),
        translation_intent_mae_mm=tuple(float(value) for value in error_mm.mean(dim=0).tolist()),
        close_emissions=int(close_emissions),
        open_emissions=int(open_emissions),
    )


def grip_release_ready_contract(split: M11IntentSplit) -> dict[str, Any]:
    """Audit the expert one-CLOSE/one-OPEN contract and pre-action OPEN gate."""
    errors: list[str] = []
    target_close_count = 0
    target_open_count = 0
    ready_open_count = 0
    episode_count = 0
    for episode_id in torch.unique(split.episode_ids, sorted=True).tolist():
        episode_count += 1
        selected = split.episode_ids == int(episode_id)
        tokens = split.tokens[selected]
        pre_ready = split.pre_release_ready[selected]
        close_indices = torch.nonzero(
            tokens[:, 3] == M9B_CLOSE_TOKEN, as_tuple=False
        ).flatten()
        open_indices = torch.nonzero(
            tokens[:, 3] == M9B_OPEN_TOKEN, as_tuple=False
        ).flatten()
        target_close_count += int(close_indices.numel())
        target_open_count += int(open_indices.numel())
        if close_indices.numel() != 1 or open_indices.numel() != 1:
            errors.append(f"episode {episode_id}: expected one CLOSE and one OPEN")
            continue
        if int(close_indices[0].item()) >= int(open_indices[0].item()):
            errors.append(f"episode {episode_id}: CLOSE must precede OPEN")
        open_index = int(open_indices[0].item())
        if bool(pre_ready[open_index].item()):
            ready_open_count += 1
        else:
            errors.append(
                f"episode {episode_id}: expert OPEN is not pre-action release-ready"
            )
    return {
        "episode_count": episode_count,
        "target_close_events": target_close_count,
        "target_open_events": target_open_count,
        "ready_gated_open_events": ready_open_count,
        "passed": not errors,
        "errors": errors,
    }


def evaluate_hybrid_split(
    actor: M11HybridTokenActor,
    split: M11IntentSplit,
    device: str | torch.device,
) -> M11HybridEvaluation:
    """Evaluate whole episodes with the exact D4 execution-time GRIP FSM."""
    target_device = torch.device(device)
    episodes: list[tuple[int, Tensor, Tensor, Tensor, Tensor]] = []
    for episode_id in torch.unique(split.episode_ids, sorted=True).tolist():
        selected = split.episode_ids == int(episode_id)
        episodes.append(
            (
                int(episode_id),
                split.observations[selected],
                split.ideal_translation_normalized[selected],
                split.tokens[selected],
                split.pre_release_ready[selected],
            )
        )
    if not episodes:
        raise RuntimeError("hybrid evaluation split contains no episodes")
    num_episodes = len(episodes)
    max_steps = max(int(tokens.shape[0]) for _id, _obs, _intent, tokens, _ready in episodes)
    lengths = torch.tensor(
        [tokens.shape[0] for _id, _obs, _intent, tokens, _ready in episodes],
        dtype=torch.long,
        device=target_device,
    )
    valid = torch.arange(max_steps, device=target_device).unsqueeze(0) < lengths.unsqueeze(1)
    observations = torch.zeros(
        num_episodes,
        max_steps,
        actor.config.observation_dim,
        dtype=torch.float32,
        device=target_device,
    )
    ready = torch.zeros(num_episodes, max_steps, dtype=torch.bool, device=target_device)
    target_intent = torch.zeros(num_episodes, max_steps, 3, dtype=torch.float32)
    target_tokens = torch.zeros(num_episodes, max_steps, 4, dtype=torch.long)
    for index, (_episode_id, obs, intent, tokens, pre_ready) in enumerate(episodes):
        length = int(tokens.shape[0])
        observations[index, :length] = obs.to(target_device)
        ready[index, :length] = pre_ready.to(target_device)
        target_intent[index, :length] = intent
        target_tokens[index, :length] = tokens

    actor.eval()
    with torch.inference_mode():
        xyz_logits, predicted_intent, raw_grip_logits, _hidden = actor.forward_sequence(
            observations
        )
        xyz_tokens = translation_tokens(xyz_logits)
        raw_grip_tokens = torch.argmax(raw_grip_logits, dim=-1)
        fsm = M11OneShotGripFSM(num_episodes, device=target_device)
        executed_grip_steps: list[Tensor] = []
        for step in range(max_steps):
            grip_tokens, _grip_execution_logits = fsm.step(
                raw_grip_logits[:, step],
                ready[:, step],
                active_mask=valid[:, step],
            )
            executed_grip_steps.append(grip_tokens)
        executed_grip = torch.stack(executed_grip_steps, dim=1)
        executed_tokens = torch.cat((xyz_tokens, executed_grip.unsqueeze(-1)), dim=-1)
        raw_tokens = torch.cat((xyz_tokens, raw_grip_tokens.unsqueeze(-1)), dim=-1)

    valid_cpu = valid.cpu()
    predicted_cpu = executed_tokens.cpu()
    raw_cpu = raw_tokens.cpu()
    intent_cpu = predicted_intent.cpu()
    metrics = hybrid_metrics(
        predicted_cpu[valid_cpu],
        target_tokens[valid_cpu],
        intent_cpu[valid_cpu],
        target_intent[valid_cpu],
        target_translation_scale_m=actor.config.target_translation_scale_m,
        close_emissions=int(fsm.close_emissions.sum().item()),
        open_emissions=int(fsm.open_emissions.sum().item()),
    )
    raw_metrics = hybrid_metrics(
        raw_cpu[valid_cpu],
        target_tokens[valid_cpu],
        intent_cpu[valid_cpu],
        target_intent[valid_cpu],
        target_translation_scale_m=actor.config.target_translation_scale_m,
        close_emissions=int(
            (raw_cpu[valid_cpu][:, 3] == M9B_CLOSE_TOKEN).sum().item()
        ),
        open_emissions=int(
            (raw_cpu[valid_cpu][:, 3] == M9B_OPEN_TOKEN).sum().item()
        ),
    )

    predicted_by_episode: list[Tensor] = []
    targets_by_episode: list[Tensor] = []
    for index, (_episode_id, _obs, _intent, tokens, _pre_ready) in enumerate(episodes):
        length = int(tokens.shape[0])
        predicted_by_episode.append(predicted_cpu[index, :length].clone())
        targets_by_episode.append(tokens.clone())
    release_contract = grip_release_ready_contract(split)
    return M11HybridEvaluation(
        metrics=metrics,
        raw_grip_metrics=raw_metrics,
        fsm_summary=fsm.summary(),
        release_ready_contract=release_contract,
        episode_ids=tuple(item[0] for item in episodes),
        predicted_tokens_by_episode=tuple(predicted_by_episode),
        target_tokens_by_episode=tuple(targets_by_episode),
    )


def d4_selection_score(metrics: M11HybridMetrics) -> float:
    return float(
        2.0 * (1.0 - metrics.exact_action_accuracy)
        + sum(1.0 - value for value in metrics.factor_accuracy[:3])
        + (1.0 - metrics.grip_open_recall)
        + (1.0 - metrics.grip_close_recall)
        + 0.1 * sum(metrics.translation_intent_mae_mm) / 3.0
    )


def d4_checkpoint_payload(
    actor: M11HybridTokenActor,
    *,
    epoch: int,
    metrics: M11HybridMetrics,
    selection_score: float,
    dataset: M11IntentDatasetBundle,
    training_config: Mapping[str, Any],
    d3_checkpoint: str,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": M11_D4_CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "validation_metrics": asdict(metrics),
        "selection_score": float(selection_score),
        "training_config": dict(training_config),
        "initialized_from_d3_checkpoint": str(d3_checkpoint),
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
        },
        "grip_fsm_contract": {
            "schema": M11_D4_GRIP_FSM_SCHEMA,
            "initial_state": "OPEN",
            "maximum_close_emissions_per_episode": 1,
            "maximum_open_emissions_per_episode": 1,
            "require_release_ready_for_open": True,
        },
        "execution_contract": {
            "direct_xyz_factorwise_argmax": True,
            "continuous_intent_is_auxiliary_only": True,
            "previous_token_input": False,
        },
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    return payload


def load_m11_d4_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11HybridTokenActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D4_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D4 checkpoint schema")
    fsm = payload.get("grip_fsm_contract", {})
    if fsm.get("schema") != M11_D4_GRIP_FSM_SCHEMA:
        raise RuntimeError("M11-B-D4 GRIP FSM contract mismatch")
    if fsm.get("require_release_ready_for_open") is not True:
        raise RuntimeError("M11-B-D4 ready-gated OPEN contract mismatch")
    raw = payload["model_config"]
    config = M11HybridModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(int(v) for v in raw["observation_hidden_dims"]),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        target_translation_scale_m=float(raw["target_translation_scale_m"]),
        category_counts=tuple(int(v) for v in raw["category_counts"]),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    actor = M11HybridTokenActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
