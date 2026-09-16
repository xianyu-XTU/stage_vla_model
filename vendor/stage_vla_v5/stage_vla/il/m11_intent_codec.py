"""M11-B-D3 continuous intent actor with an explicit error-feedback codec.

Q1 demonstrations were produced by quantizing a smooth metric substep with a
stateful residual.  D2 asked a classifier to infer that hidden codec state from
token history.  D3 restores the original causal decomposition: the network
predicts the pre-quantized metric intent and this module owns the residual and
the unchanged five-bin XYZ Action DSL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from stage_vla.rl.action_dsl import (
    M9B_CATEGORY_COUNTS,
    M9B_CLOSE_TOKEN,
    M9B_OPEN_TOKEN,
    M9B_TRANSLATION_CENTER_TOKEN,
    M9B_TRANSLATION_MAX_ABS_BIN,
)

from .m11_behavior_cloning import M11BCDatasetBundle, M11BCSplit, load_m11_bc_dataset
from .m11_demonstrations import M11_QUANTIZER_METHOD, validate_m7_trace_payload

M11_D3_CHECKPOINT_SCHEMA = "stage_vla.m11.intent_codec_actor.v1"
M11_D3_SUMMARY_SCHEMA = "stage_vla.m11.intent_codec_summary.v1"
M11_D3_CODEC_SCHEMA = "stage_vla.m11.error_feedback_codec.v1"
M11_D3_FACTOR_NAMES = ("dx", "dy", "dz", "grip")


@dataclass(frozen=True)
class M11IntentModelConfig:
    observation_dim: int = 99
    observation_hidden_dims: tuple[int, ...] = (256, 256)
    recurrent_hidden_dim: int = 256
    head_hidden_dim: int = 128
    target_translation_scale_m: float = 0.004
    category_counts: tuple[int, ...] = M9B_CATEGORY_COUNTS
    activation: str = "elu"
    observation_normalization: bool = False

    def __post_init__(self) -> None:
        if self.observation_dim != 99:
            raise ValueError("M11-B-D3 requires the exact 99-D F1 observation")
        if tuple(self.category_counts) != tuple(M9B_CATEGORY_COUNTS):
            raise ValueError("M11-B-D3 must preserve the [5,5,5,3] Action DSL")
        dimensions = (*self.observation_hidden_dims, self.recurrent_hidden_dim, self.head_hidden_dim)
        if not self.observation_hidden_dims or any(int(value) <= 0 for value in dimensions):
            raise ValueError("all hidden dimensions must be positive")
        if not math.isclose(self.target_translation_scale_m, 0.004, abs_tol=1.0e-12):
            raise ValueError("M11-B-D3 preserves the F1 4 mm translation scale")
        if self.activation.lower() != "elu":
            raise ValueError("M11-B-D3 requires ELU")
        if self.observation_normalization:
            raise ValueError("M11-B-D3 keeps F1 observation_normalization=False")


class M11IntentCodecActor(nn.Module):
    """Causal observation encoder + GRU with metric-intent and GRIP heads."""

    def __init__(self, config: M11IntentModelConfig) -> None:
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
        self.translation_head = nn.Linear(config.head_hidden_dim, 3)
        self.grip_head = nn.Linear(config.head_hidden_dim, 3)

    def forward_sequence(
        self,
        observations: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 3 or value.shape[-1] != self.config.observation_dim:
            raise ValueError("observations must have shape [B,T,99]")
        if not torch.isfinite(value).all():
            raise ValueError("observations contain NaN/Inf")
        encoded = self.observation_encoder(value)
        recurrent, next_hidden = self.gru(encoded, hidden)
        latent = self.shared_head(recurrent)
        # Normalized metric intent: -1/+1 are exactly -/+4 mm.
        intent_normalized = torch.tanh(self.translation_head(latent))
        return intent_normalized, self.grip_head(latent), next_hidden

    def step(
        self,
        observations: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        value = torch.as_tensor(observations)
        if value.ndim != 2:
            raise ValueError("step expects observations [N,99]")
        intent, grip_logits, next_hidden = self.forward_sequence(value.unsqueeze(1), hidden)
        return intent[:, 0], grip_logits[:, 0], next_hidden


class M11ErrorFeedbackCodec:
    """Stateful exact five-bin XYZ codec used at training audit and deployment."""

    def __init__(self, num_envs: int, *, device: str | torch.device = "cpu") -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.residual_normalized = torch.zeros(self.num_envs, 3, device=self.device)
        self._max_abs_residual_normalized = torch.zeros((), device=self.device)

    @property
    def max_abs_residual_normalized(self) -> float:
        return float(self._max_abs_residual_normalized.item())

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            self.residual_normalized.zero_()
            self._max_abs_residual_normalized.zero_()
            return
        selected = torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        if selected.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [num_envs]")
        self.residual_normalized[selected] = 0.0

    def step(
        self,
        intent_normalized: Tensor,
        grip_logits: Tensor,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        intent = torch.as_tensor(intent_normalized, device=self.device)
        grip = torch.as_tensor(grip_logits, device=self.device)
        if intent.shape != (self.num_envs, 3) or grip.shape != (self.num_envs, 3):
            raise ValueError("codec expects intent [N,3] and grip_logits [N,3]")
        if not torch.isfinite(intent).all() or not torch.isfinite(grip).all():
            raise ValueError("codec input contains NaN/Inf")
        if active_mask is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            active = torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1)
            if active.shape != (self.num_envs,):
                raise ValueError("active_mask must have shape [num_envs]")
        requested = intent.clamp(-1.0, 1.0) + self.residual_normalized
        signed_bins = torch.round(requested.clamp(-1.0, 1.0) * 2.0).to(torch.long)
        signed_bins.clamp_(-M9B_TRANSLATION_MAX_ABS_BIN, M9B_TRANSLATION_MAX_ABS_BIN)
        translation_tokens = signed_bins + M9B_TRANSLATION_CENTER_TOKEN
        executed = signed_bins.to(intent.dtype) / float(M9B_TRANSLATION_MAX_ABS_BIN)
        next_residual = requested - executed
        self.residual_normalized = torch.where(
            active.unsqueeze(-1), next_residual, self.residual_normalized
        )
        # Callers never advance a fully inactive batch; keeping this branch
        # tensor-only avoids one GPU synchronization per control step.
        self._max_abs_residual_normalized = torch.maximum(
            self._max_abs_residual_normalized,
            self.residual_normalized[active].abs().amax(),
        )
        grip_token = torch.argmax(grip, dim=-1)
        tokens = torch.cat((translation_tokens, grip_token.unsqueeze(-1)), dim=-1)
        # Exact execution logits make the diagnostic argmax agree with the
        # stateful codec even at round-to-even half-bin ties.
        translation_logits: list[Tensor] = []
        for factor in range(3):
            part = torch.full((self.num_envs, 5), -20.0, device=self.device, dtype=grip.dtype)
            part.scatter_(1, translation_tokens[:, factor : factor + 1], 20.0)
            translation_logits.append(part)
        return tokens, torch.cat((*translation_logits, grip), dim=-1)


@dataclass(frozen=True)
class M11IntentSplit:
    name: str
    observations: Tensor
    tokens: Tensor
    ideal_translation_normalized: Tensor
    residual_before_normalized: Tensor
    pre_release_ready: Tensor
    stage_ids: Tensor
    episode_step_ids: Tensor
    episode_ids: Tensor
    episode_seeds: tuple[int, ...]
    episode_count: int
    step_count: int
    observation_dim: int
    minimum_episode_steps: int
    maximum_episode_steps: int


@dataclass(frozen=True)
class M11IntentDatasetBundle:
    dataset_dir: str
    manifest_sha256: str
    quantizer_method: str
    train: M11IntentSplit
    validation: M11IntentSplit


@dataclass(frozen=True)
class M11IntentMetrics:
    num_steps: int
    translation_mae_mm: tuple[float, float, float]
    translation_rmse_mm: tuple[float, float, float]
    factor_accuracy: tuple[float, float, float, float]
    exact_action_accuracy: float
    grip_open_recall: float
    grip_close_recall: float
    maximum_codec_residual_mm: float


def _trace_path(dataset_root: Path, episode: Mapping[str, Any]) -> Path:
    source_text = str(episode.get("source_trace", ""))
    original = Path(source_text)
    # A dataset copied from Windows can be audited on another platform; do not
    # let POSIX treat backslashes as literal filename characters.
    portable_name = source_text.replace("\\", "/").rsplit("/", 1)[-1]
    seed = int(episode.get("seed", -1))
    canonical_name = f"seed_{seed:08d}.json" if seed >= 0 else ""
    candidates = (
        original,
        dataset_root / "_work" / "traces" / portable_name,
        dataset_root / "_work" / "traces" / canonical_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "M11-B-D3 requires the original M11-A source trace; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def ideal_translation_targets(
    trace_payload: Mapping[str, Any],
    source_step_indices: Tensor,
    reference_tokens: Tensor,
    *,
    target_translation_scale_m: float,
) -> tuple[Tensor, Tensor]:
    """Recover exact pre-quantized substep intent and residual-before state."""
    validate_m7_trace_payload(trace_payload)
    source_indices = torch.as_tensor(source_step_indices, dtype=torch.long).reshape(-1)
    tokens = torch.as_tensor(reference_tokens, dtype=torch.long)
    if tokens.shape != (source_indices.numel(), 4):
        raise ValueError("source_step_indices and tokens are not aligned")
    records = trace_payload["records"]
    if bool(((source_indices < 0) | (source_indices >= len(records))).any().item()):
        raise ValueError("source_step_indices reference a missing trace record")
    source_scale = torch.tensor(trace_payload["source_action_scale_xyz"], dtype=torch.float64)
    per_source: list[Tensor] = []
    required_counts: list[int] = []
    for record in records:
        raw = torch.tensor(record["raw_action"], dtype=torch.float64)
        processed_m = raw[:3] * source_scale
        required = int(
            max(
                1,
                math.ceil(
                    float(processed_m.abs().max().item())
                    / float(target_translation_scale_m)
                    - 1.0e-12
                ),
            )
        )
        per_source.append(processed_m / float(required) / float(target_translation_scale_m))
        required_counts.append(required)
    table = torch.stack(per_source)
    ideal = table[source_indices]
    # Every recorded source action must expand to exactly the same number of
    # substeps as the accepted Q1 episode.
    for source_id in torch.unique(source_indices, sorted=True).tolist():
        actual = int((source_indices == int(source_id)).sum().item())
        if actual != required_counts[int(source_id)]:
            raise RuntimeError(
                f"source step {source_id} expected {required_counts[int(source_id)]} "
                f"Q1 substeps, found {actual}"
            )
    residual_before = torch.zeros_like(ideal)
    residual = torch.zeros(3, dtype=torch.float64)
    for step in range(tokens.shape[0]):
        residual_before[step] = residual
        executed = (
            tokens[step, :3].to(torch.float64) - M9B_TRANSLATION_CENTER_TOKEN
        ) / float(M9B_TRANSLATION_MAX_ABS_BIN)
        requested = ideal[step] + residual
        reconstructed = torch.round(requested.clamp(-1.0, 1.0) * 2.0).to(torch.long)
        reconstructed = reconstructed + M9B_TRANSLATION_CENTER_TOKEN
        if not torch.equal(reconstructed, tokens[step, :3]):
            raise RuntimeError(f"source trace cannot reconstruct accepted Q1 token at step {step}")
        residual = requested - executed
        if float(residual.abs().max().item()) > 0.250001:
            raise RuntimeError("reference Q1 residual exceeded half a bin")
    return ideal.to(torch.float32), residual_before.to(torch.float32)


def _augment_split(
    root: Path,
    manifest: Mapping[str, Any],
    base: M11BCSplit,
) -> M11IntentSplit:
    ideals: list[Tensor] = []
    residuals: list[Tensor] = []
    pre_ready_values: list[Tensor] = []
    stage_values: list[Tensor] = []
    episode_step_values: list[Tensor] = []
    lengths: list[int] = []
    entries = [entry for entry in manifest["episodes"] if entry.get("split") == base.name]
    offset = 0
    for entry in entries:
        episode_path = (root / str(entry["path"])).resolve()
        if root != episode_path and root not in episode_path.parents:
            raise ValueError("episode path escapes dataset root")
        episode = torch.load(episode_path, map_location="cpu", weights_only=False)
        trace_path = _trace_path(root, episode)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if int(trace.get("seed", -1)) != int(episode["seed"]):
            raise RuntimeError("source trace and accepted episode seed mismatch")
        steps = int(entry["steps"])
        expected_tokens = base.tokens[offset : offset + steps]
        episode_tokens = torch.as_tensor(episode["token_indices"], dtype=torch.long)
        if not torch.equal(expected_tokens, episode_tokens):
            raise RuntimeError("base dataset and intent episode tokens disagree")
        target_scale = float(episode["quantization"]["target_translation_scale_m"])
        if not math.isclose(target_scale, 0.004, abs_tol=1.0e-12):
            raise RuntimeError("accepted Q1 episode does not use the F1 4 mm target scale")
        ideal, residual = ideal_translation_targets(
            trace,
            episode["source_step_indices"],
            episode_tokens,
            target_translation_scale_m=target_scale,
        )
        executed_last = (
            episode_tokens[-1, :3].to(torch.float32) - M9B_TRANSLATION_CENTER_TOKEN
        ) / float(M9B_TRANSLATION_MAX_ABS_BIN)
        reconstructed_final = residual[-1] + ideal[-1] - executed_last
        recorded_final = torch.tensor(
            episode["quantization"]["final_translation_residual_m"], dtype=torch.float32
        ) / target_scale
        if not torch.allclose(reconstructed_final, recorded_final, atol=1.0e-5, rtol=0.0):
            raise RuntimeError("source trace final residual disagrees with accepted Q1 metadata")
        ideals.append(ideal)
        residuals.append(residual)
        post_ready = torch.as_tensor(
            episode.get("truth", {}).get("release_ready"), dtype=torch.bool
        ).reshape(-1)
        stage_ids = torch.as_tensor(episode.get("stage_ids"), dtype=torch.long).reshape(-1)
        if post_ready.shape != (steps,) or stage_ids.shape != (steps,):
            raise RuntimeError("accepted episode lacks aligned release-ready/stage metadata")
        pre_ready_values.append(
            torch.cat((torch.zeros(1, dtype=torch.bool), post_ready[:-1]), dim=0)
        )
        stage_values.append(stage_ids)
        episode_step_values.append(torch.arange(steps, dtype=torch.long))
        lengths.append(steps)
        offset += steps
    if offset != base.step_count:
        raise RuntimeError("intent metadata does not cover the full split")
    return M11IntentSplit(
        name=base.name,
        observations=base.observations,
        tokens=base.tokens,
        ideal_translation_normalized=torch.cat(ideals),
        residual_before_normalized=torch.cat(residuals),
        pre_release_ready=torch.cat(pre_ready_values),
        stage_ids=torch.cat(stage_values),
        episode_step_ids=torch.cat(episode_step_values),
        episode_ids=base.episode_ids,
        episode_seeds=base.episode_seeds,
        episode_count=base.episode_count,
        step_count=base.step_count,
        observation_dim=base.observation_dim,
        minimum_episode_steps=min(lengths),
        maximum_episode_steps=max(lengths),
    )


def load_m11_intent_dataset(
    dataset_dir: str | Path,
    *,
    minimum_train_episodes: int = 100,
    minimum_validation_episodes: int = 20,
) -> M11IntentDatasetBundle:
    base: M11BCDatasetBundle = load_m11_bc_dataset(
        dataset_dir,
        minimum_train_episodes=minimum_train_episodes,
        minimum_validation_episodes=minimum_validation_episodes,
    )
    root = Path(base.dataset_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("quantizer_method") != M11_QUANTIZER_METHOD:
        raise RuntimeError("M11-B-D3 requires the Q1 error-feedback dataset")
    return M11IntentDatasetBundle(
        dataset_dir=base.dataset_dir,
        manifest_sha256=base.manifest_sha256,
        quantizer_method=base.quantizer_method,
        train=_augment_split(root, manifest, base.train),
        validation=_augment_split(root, manifest, base.validation),
    )


def soft_translation_token_logits(
    intent_normalized: Tensor,
    residual_before_normalized: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    intent = torch.as_tensor(intent_normalized)
    residual = torch.as_tensor(residual_before_normalized, device=intent.device, dtype=intent.dtype)
    if intent.shape != residual.shape or intent.shape[-1] != 3:
        raise ValueError("intent and residual must have matching [...,3] shapes")
    centers = torch.linspace(-1.0, 1.0, 5, device=intent.device, dtype=intent.dtype)
    requested = intent + residual
    parts = [
        -((requested[..., factor : factor + 1] - centers) / temperature).square()
        for factor in range(3)
    ]
    return torch.cat(parts, dim=-1)


def intent_codec_loss(
    intent_normalized: Tensor,
    grip_logits: Tensor,
    target_intent_normalized: Tensor,
    residual_before_normalized: Tensor,
    target_tokens: Tensor,
    mask: Tensor,
    *,
    token_temperature: float,
    translation_regression_weight: float,
    translation_token_weight: float,
    grip_weight: float,
    grip_transition_step_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid = torch.as_tensor(mask, dtype=torch.bool, device=intent_normalized.device)
    target_intent = torch.as_tensor(
        target_intent_normalized, device=intent_normalized.device, dtype=intent_normalized.dtype
    )
    target = torch.as_tensor(target_tokens, device=intent_normalized.device, dtype=torch.long)
    if intent_normalized.shape != target_intent.shape or intent_normalized.shape[-1] != 3:
        raise ValueError("intent targets must align as [B,T,3]")
    if grip_logits.shape != (*valid.shape, 3) or target.shape != (*valid.shape, 4):
        raise ValueError("grip/tokens/mask shapes are not aligned")
    if not bool(valid.any().item()):
        raise ValueError("loss mask contains no target steps")
    regression = F.smooth_l1_loss(intent_normalized[valid], target_intent[valid])
    translation_logits = soft_translation_token_logits(
        intent_normalized,
        residual_before_normalized,
        temperature=token_temperature,
    )
    translation_losses = []
    for factor, part in enumerate(torch.split(translation_logits, 5, dim=-1)):
        per_step = F.cross_entropy(
            part.reshape(-1, 5), target[..., factor].reshape(-1), reduction="none"
        ).reshape(valid.shape)
        translation_losses.append(per_step[valid].mean())
    translation_token = torch.stack(translation_losses).mean()
    grip_per_step = F.cross_entropy(
        grip_logits.reshape(-1, 3), target[..., 3].reshape(-1), reduction="none"
    ).reshape(valid.shape)
    transition = (target[..., 3] == M9B_OPEN_TOKEN) | (target[..., 3] == M9B_CLOSE_TOKEN)
    step_weights = torch.where(
        transition,
        torch.full_like(grip_per_step, float(grip_transition_step_weight)),
        torch.ones_like(grip_per_step),
    )
    grip_loss = (grip_per_step[valid] * step_weights[valid]).sum() / step_weights[valid].sum()
    total = (
        float(translation_regression_weight) * regression
        + float(translation_token_weight) * translation_token
        + float(grip_weight) * grip_loss
    )
    return total, {
        "translation_regression": regression,
        "translation_token_ce": translation_token,
        "grip_ce": grip_loss,
    }


def intent_metrics(
    predicted_intent_normalized: Tensor,
    target_intent_normalized: Tensor,
    predicted_tokens: Tensor,
    target_tokens: Tensor,
    *,
    target_translation_scale_m: float,
    maximum_codec_residual_normalized: float,
) -> M11IntentMetrics:
    predicted_intent = torch.as_tensor(predicted_intent_normalized).cpu()
    target_intent = torch.as_tensor(target_intent_normalized).cpu()
    predicted = torch.as_tensor(predicted_tokens, dtype=torch.long).cpu()
    target = torch.as_tensor(target_tokens, dtype=torch.long).cpu()
    if predicted_intent.shape != target_intent.shape or predicted.shape != target.shape:
        raise ValueError("metric predictions and targets must align")
    error_mm = (predicted_intent - target_intent) * float(target_translation_scale_m) * 1000.0
    factor_accuracy = tuple(
        float((predicted[:, factor] == target[:, factor]).to(torch.float64).mean().item())
        for factor in range(4)
    )
    exact = float((predicted == target).all(dim=-1).to(torch.float64).mean().item())
    grip = target[:, 3]
    predicted_grip = predicted[:, 3]

    def recall(token: int) -> float:
        selected = grip == int(token)
        if not bool(selected.any().item()):
            return 0.0
        return float((predicted_grip[selected] == int(token)).to(torch.float64).mean().item())

    return M11IntentMetrics(
        num_steps=int(target.shape[0]),
        translation_mae_mm=tuple(float(v) for v in error_mm.abs().mean(dim=0).tolist()),
        translation_rmse_mm=tuple(float(v) for v in error_mm.square().mean(dim=0).sqrt().tolist()),
        factor_accuracy=factor_accuracy,
        exact_action_accuracy=exact,
        grip_open_recall=recall(M9B_OPEN_TOKEN),
        grip_close_recall=recall(M9B_CLOSE_TOKEN),
        maximum_codec_residual_mm=(
            float(maximum_codec_residual_normalized)
            * float(target_translation_scale_m)
            * 1000.0
        ),
    )


def d3_selection_score(metrics: M11IntentMetrics) -> float:
    return float(
        sum(metrics.translation_mae_mm) / 3.0
        + 2.0 * (1.0 - metrics.exact_action_accuracy)
        + 0.5 * (2.0 - metrics.grip_open_recall - metrics.grip_close_recall)
    )


def d3_checkpoint_payload(
    actor: M11IntentCodecActor,
    *,
    epoch: int,
    metrics: M11IntentMetrics,
    selection_score: float,
    dataset: M11IntentDatasetBundle,
    training_config: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": M11_D3_CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "validation_metrics": asdict(metrics),
        "selection_score": float(selection_score),
        "training_config": dict(training_config),
        "dataset": {
            "directory": dataset.dataset_dir,
            "manifest_sha256": dataset.manifest_sha256,
            "quantizer_method": dataset.quantizer_method,
            "train_episodes": dataset.train.episode_count,
            "validation_episodes": dataset.validation.episode_count,
            "train_steps": dataset.train.step_count,
            "validation_steps": dataset.validation.step_count,
            "minimum_episode_steps": min(
                dataset.train.minimum_episode_steps, dataset.validation.minimum_episode_steps
            ),
            "maximum_episode_steps": max(
                dataset.train.maximum_episode_steps, dataset.validation.maximum_episode_steps
            ),
        },
        "codec_contract": {
            "schema": M11_D3_CODEC_SCHEMA,
            "initial_residual_normalized": [0.0, 0.0, 0.0],
            "residual_reset_on_episode_reset": True,
            "translation_bins_normalized": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "rounding": "torch.round_ties_to_even",
        },
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    return payload


def load_m11_d3_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11IntentCodecActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D3_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D3 checkpoint schema")
    codec = payload.get("codec_contract", {})
    if codec.get("schema") != M11_D3_CODEC_SCHEMA:
        raise RuntimeError("M11-B-D3 codec contract mismatch")
    if codec.get("initial_residual_normalized") != [0.0, 0.0, 0.0]:
        raise RuntimeError("M11-B-D3 initial residual contract mismatch")
    raw = payload["model_config"]
    config = M11IntentModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(int(v) for v in raw["observation_hidden_dims"]),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        target_translation_scale_m=float(raw["target_translation_scale_m"]),
        category_counts=tuple(int(v) for v in raw["category_counts"]),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    actor = M11IntentCodecActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
