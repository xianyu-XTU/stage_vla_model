"""M11-B offline behavior cloning for the factorized Stage-VLA actor.

The actor intentionally mirrors the F1 RSL-RL MLP topology while remaining a
plain PyTorch module.  It consumes the exact 99-D policy observations captured
by accepted M11-A same-seed replays and emits concatenated logits for the
existing ``[5,5,5,3]`` Action-DSL factors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from stage_vla.rl.action_dsl import M9B_CATEGORY_COUNTS
from stage_vla.rl.factorized_categorical_core import (
    deterministic_factor_tokens,
    split_factor_logits,
    validate_factor_tokens,
)

from .m11_demonstrations import (
    M11_DATASET_SCHEMA,
    M11_QUANTIZER_METHOD,
)

M11_BC_CHECKPOINT_SCHEMA = "stage_vla.m11.factorized_bc_actor.v1"
M11_BC_SUMMARY_SCHEMA = "stage_vla.m11.factorized_bc_summary.v1"
M11_BC_FACTOR_NAMES = ("dx", "dy", "dz", "grip")


@dataclass(frozen=True)
class M11BCModelConfig:
    observation_dim: int
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    category_counts: tuple[int, ...] = M9B_CATEGORY_COUNTS
    activation: str = "elu"
    observation_normalization: bool = False

    def __post_init__(self) -> None:
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be > 0")
        if not self.hidden_dims or any(int(value) <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        if tuple(int(v) for v in self.category_counts) != tuple(M9B_CATEGORY_COUNTS):
            raise ValueError(
                f"M11-B must preserve Action-DSL counts {M9B_CATEGORY_COUNTS}"
            )
        if self.activation.lower() != "elu":
            raise ValueError("M11-B currently requires ELU to match the F1 actor")
        if self.observation_normalization:
            raise ValueError("F1 uses obs_normalization=False; M11-B must match it")


class M11FactorizedActor(nn.Module):
    """F1-compatible MLP producing 18 concatenated categorical logits."""

    def __init__(self, config: M11BCModelConfig) -> None:
        super().__init__()
        self.config = config
        widths = (config.observation_dim, *config.hidden_dims, sum(config.category_counts))
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(nn.ELU())
        # The name ``mlp`` and Sequential key layout intentionally match the
        # RSL-RL actor surface used by existing M10 evaluators.
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        value = torch.as_tensor(observations)
        if value.ndim != 2 or value.shape[-1] != self.config.observation_dim:
            raise ValueError(
                "observations must have shape [N,observation_dim], got "
                f"{tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("observations contain NaN/Inf")
        logits = self.mlp(value)
        if logits.shape != (value.shape[0], sum(self.config.category_counts)):
            raise RuntimeError(f"unexpected actor logit shape {tuple(logits.shape)}")
        return logits

    def deterministic_tokens(self, observations: Tensor) -> Tensor:
        return deterministic_factor_tokens(
            self(observations), self.config.category_counts
        )


@dataclass(frozen=True)
class M11BCSplit:
    name: str
    observations: Tensor
    tokens: Tensor
    episode_ids: Tensor
    episode_seeds: tuple[int, ...]
    episode_count: int
    step_count: int
    observation_dim: int


@dataclass(frozen=True)
class M11BCDatasetBundle:
    dataset_dir: str
    manifest_sha256: str
    quantizer_method: str
    train: M11BCSplit
    validation: M11BCSplit


@dataclass(frozen=True)
class M11BCMetrics:
    num_steps: int
    joint_nll: float
    factor_nll: tuple[float, ...]
    factor_accuracy: tuple[float, ...]
    factor_macro_recall: tuple[float, ...]
    exact_action_accuracy: float
    grip_open_recall: float
    grip_close_recall: float
    confusion_matrices: tuple[tuple[tuple[int, ...], ...], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_episode_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"episode path escapes dataset root: {relative!r}")
    return candidate


def _load_split(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    split: str,
    verify_hashes: bool,
) -> M11BCSplit:
    observations: list[Tensor] = []
    tokens: list[Tensor] = []
    episode_ids: list[Tensor] = []
    seeds: list[int] = []
    observation_dim: int | None = None

    selected = [entry for entry in entries if entry.get("split") == split]
    if not selected:
        raise RuntimeError(f"M11-B dataset contains no {split!r} episodes")

    for entry in selected:
        episode_id = int(entry["episode_id"])
        episode_path = _safe_episode_path(root, str(entry["path"]))
        if not episode_path.is_file():
            raise FileNotFoundError(episode_path)
        if verify_hashes and _sha256(episode_path) != str(entry["sha256"]):
            raise RuntimeError(f"episode SHA-256 mismatch: {episode_path}")
        episode = torch.load(episode_path, map_location="cpu", weights_only=False)
        if episode.get("schema") != M11_DATASET_SCHEMA:
            raise RuntimeError(f"episode schema mismatch: {episode_path}")
        if episode.get("split") != split:
            raise RuntimeError(f"episode/manifest split mismatch: {episode_path}")
        if int(episode.get("episode_id", -1)) != episode_id:
            raise RuntimeError(f"episode id mismatch: {episode_path}")
        if int(episode.get("seed", -1)) != int(entry["seed"]):
            raise RuntimeError(f"episode seed mismatch: {episode_path}")
        if episode.get("quantization", {}).get("method") != M11_QUANTIZER_METHOD:
            raise RuntimeError(f"episode is not Q1-quantized: {episode_path}")
        if int(episode.get("strict_summary", {}).get("strict_successes", 0)) != 1:
            raise RuntimeError(f"episode lacks unchanged strict success: {episode_path}")

        obs = torch.as_tensor(episode["observations"], dtype=torch.float32)
        action = validate_factor_tokens(
            torch.as_tensor(episode["token_indices"]), M9B_CATEGORY_COUNTS
        ).cpu()
        if obs.ndim != 2 or action.ndim != 2 or obs.shape[0] != action.shape[0]:
            raise RuntimeError(f"observation/action alignment mismatch: {episode_path}")
        if int(entry.get("steps", -1)) != int(obs.shape[0]):
            raise RuntimeError(f"episode step count mismatch: {episode_path}")
        if not torch.isfinite(obs).all():
            raise RuntimeError(f"non-finite observation: {episode_path}")
        if observation_dim is None:
            observation_dim = int(obs.shape[1])
        elif int(obs.shape[1]) != observation_dim:
            raise RuntimeError(f"observation dimension mismatch: {episode_path}")

        observations.append(obs.contiguous())
        tokens.append(action.to(torch.long).contiguous())
        episode_ids.append(torch.full((obs.shape[0],), episode_id, dtype=torch.long))
        seeds.append(int(episode["seed"]))

    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"duplicate seed inside {split} split")
    merged_obs = torch.cat(observations, dim=0)
    merged_tokens = torch.cat(tokens, dim=0)
    merged_episode_ids = torch.cat(episode_ids, dim=0)
    return M11BCSplit(
        name=split,
        observations=merged_obs,
        tokens=merged_tokens,
        episode_ids=merged_episode_ids,
        episode_seeds=tuple(seeds),
        episode_count=len(selected),
        step_count=int(merged_obs.shape[0]),
        observation_dim=int(observation_dim),
    )


def load_m11_bc_dataset(
    dataset_dir: str | Path,
    *,
    verify_hashes: bool = True,
    minimum_train_episodes: int = 100,
    minimum_validation_episodes: int = 20,
) -> M11BCDatasetBundle:
    """Load audited M11-A episodes without leaking trajectories across splits."""
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "stage_vla.m11.dsl_dataset_manifest.v1":
        raise RuntimeError("M11-A manifest schema mismatch")
    if not bool(manifest.get("complete")):
        raise RuntimeError("M11-A manifest is not complete")
    if manifest.get("quantizer_method") != M11_QUANTIZER_METHOD:
        raise RuntimeError("M11-B requires the Q1 error-feedback dataset")
    entries = manifest.get("episodes")
    required_episodes = minimum_train_episodes + minimum_validation_episodes
    if not isinstance(entries, list) or len(entries) < required_episodes:
        raise RuntimeError(
            "M11-B dataset is too small: "
            f"{0 if not isinstance(entries, list) else len(entries)} < {required_episodes}"
        )
    episode_ids = [int(entry["episode_id"]) for entry in entries]
    if len(episode_ids) != len(set(episode_ids)):
        raise RuntimeError("duplicate episode id in M11-A manifest")

    train = _load_split(root, entries, split="train", verify_hashes=verify_hashes)
    validation = _load_split(
        root, entries, split="validation", verify_hashes=verify_hashes
    )
    if train.episode_count < minimum_train_episodes:
        raise RuntimeError(
            f"train episodes {train.episode_count} < {minimum_train_episodes}"
        )
    if validation.episode_count < minimum_validation_episodes:
        raise RuntimeError(
            f"validation episodes {validation.episode_count} < {minimum_validation_episodes}"
        )
    if train.observation_dim != validation.observation_dim:
        raise RuntimeError("train/validation observation dimensions differ")
    if set(train.episode_seeds) & set(validation.episode_seeds):
        raise RuntimeError("trajectory seed leakage between train and validation")
    return M11BCDatasetBundle(
        dataset_dir=str(root),
        manifest_sha256=_sha256(manifest_path),
        quantizer_method=M11_QUANTIZER_METHOD,
        train=train,
        validation=validation,
    )


def trajectory_balanced_step_weights(episode_ids: Tensor) -> Tensor:
    """Give every trajectory equal total sampling mass."""
    ids = torch.as_tensor(episode_ids, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("episode_ids must be a non-empty vector")
    unique, inverse, counts = torch.unique(
        ids, sorted=True, return_inverse=True, return_counts=True
    )
    if unique.numel() == 0:
        raise ValueError("episode_ids contains no trajectories")
    weights = 1.0 / counts[inverse].to(torch.float64)
    return weights / weights.mean()


def factor_class_weights(
    tokens: Tensor,
    *,
    category_counts: Sequence[int] = M9B_CATEGORY_COUNTS,
    power: float = 0.5,
    max_weight: float = 10.0,
) -> tuple[Tensor, ...]:
    """Return capped inverse-frequency weights while preserving raw metrics."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("class-weight power must be inside [0,1]")
    if max_weight < 1.0:
        raise ValueError("max_weight must be >= 1")
    idx = validate_factor_tokens(tokens, category_counts).cpu()
    result: list[Tensor] = []
    for factor, count in enumerate(category_counts):
        histogram = torch.bincount(idx[:, factor], minlength=int(count)).to(torch.float64)
        present = histogram > 0
        safe = histogram.clamp_min(1.0)
        raw = (float(idx.shape[0]) / (float(count) * safe)).pow(power)
        raw = raw.clamp(max=float(max_weight))
        raw = raw / raw[present].mean()
        result.append(raw.to(torch.float32))
    return tuple(result)


def factorized_weighted_ce_loss(
    logits: Tensor,
    tokens: Tensor,
    *,
    class_weights: Sequence[Tensor] | None = None,
    category_counts: Sequence[int] = M9B_CATEGORY_COUNTS,
) -> Tensor:
    parts = split_factor_logits(logits, category_counts)
    idx = validate_factor_tokens(tokens, category_counts).to(logits.device)
    if class_weights is not None and len(class_weights) != len(parts):
        raise ValueError("class_weights length must match factor count")
    losses = []
    for factor, part in enumerate(parts):
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(
                class_weights[factor], device=part.device, dtype=part.dtype
            )
        losses.append(F.cross_entropy(part, idx[:, factor], weight=weight))
    return torch.stack(losses).mean()


def metrics_from_logits(
    logits: Tensor,
    tokens: Tensor,
    *,
    category_counts: Sequence[int] = M9B_CATEGORY_COUNTS,
) -> M11BCMetrics:
    parts = split_factor_logits(logits, category_counts)
    idx = validate_factor_tokens(tokens, category_counts).to(logits.device)
    if logits.ndim != 2 or idx.ndim != 2 or logits.shape[0] != idx.shape[0]:
        raise ValueError("logits/tokens must be aligned 2-D batches")
    num_steps = int(idx.shape[0])
    if num_steps <= 0:
        raise ValueError("metrics require at least one step")

    factor_nll: list[float] = []
    factor_accuracy: list[float] = []
    factor_macro_recall: list[float] = []
    confusion_matrices: list[tuple[tuple[int, ...], ...]] = []
    predictions: list[Tensor] = []
    for factor, (part, count) in enumerate(zip(parts, category_counts, strict=True)):
        target = idx[:, factor]
        prediction = torch.argmax(part, dim=-1)
        predictions.append(prediction)
        factor_nll.append(
            float(F.cross_entropy(part, target, reduction="mean").item())
        )
        factor_accuracy.append(float((prediction == target).float().mean().item()))
        encoded = target * int(count) + prediction
        matrix = torch.bincount(
            encoded, minlength=int(count) * int(count)
        ).reshape(int(count), int(count))
        support = matrix.sum(dim=1)
        recalls = matrix.diag().to(torch.float64) / support.clamp_min(1).to(torch.float64)
        present = support > 0
        factor_macro_recall.append(float(recalls[present].mean().item()))
        confusion_matrices.append(
            tuple(tuple(int(value) for value in row) for row in matrix.cpu().tolist())
        )

    stacked_predictions = torch.stack(predictions, dim=-1)
    exact = float((stacked_predictions == idx).all(dim=-1).float().mean().item())
    grip_matrix = confusion_matrices[3]

    def recall_for_grip(token: int) -> float:
        row = grip_matrix[token]
        support = sum(row)
        return float(row[token] / support) if support else 0.0

    return M11BCMetrics(
        num_steps=num_steps,
        joint_nll=float(sum(factor_nll)),
        factor_nll=tuple(factor_nll),
        factor_accuracy=tuple(factor_accuracy),
        factor_macro_recall=tuple(factor_macro_recall),
        exact_action_accuracy=exact,
        grip_open_recall=recall_for_grip(0),
        grip_close_recall=recall_for_grip(2),
        confusion_matrices=tuple(confusion_matrices),
    )


def checkpoint_selection_score(
    metrics: M11BCMetrics,
    *,
    transition_penalty_weight: float = 0.5,
    factor_error_weight: float = 0.05,
) -> float:
    """Lower is better: unweighted NLL plus rare-GRIP/accuracy safeguards."""
    values = [metrics.joint_nll, *metrics.factor_accuracy]
    if not all(math.isfinite(value) for value in values):
        return math.inf
    transition_penalty = (
        2.0 - metrics.grip_open_recall - metrics.grip_close_recall
    )
    factor_penalty = sum(1.0 - value for value in metrics.factor_accuracy)
    return float(
        metrics.joint_nll
        + transition_penalty_weight * transition_penalty
        + factor_error_weight * factor_penalty
    )


def checkpoint_payload(
    actor: M11FactorizedActor,
    *,
    epoch: int,
    metrics: M11BCMetrics,
    selection_score: float,
    dataset: M11BCDatasetBundle,
    training_config: Mapping[str, Any],
    class_weights: Sequence[Tensor],
    optimizer_state_dict: Mapping[str, Any] | None = None,
    scheduler_state_dict: Mapping[str, Any] | None = None,
    best_epoch: int | None = None,
    best_score: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": M11_BC_CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "actor_mlp_state_dict": actor.mlp.state_dict(),
        "validation_metrics": asdict(metrics),
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
            "train_seeds": list(dataset.train.episode_seeds),
            "validation_seeds": list(dataset.validation.episode_seeds),
        },
        "rsl_actor_compatibility": {
            "target": "actor.mlp.load_state_dict(actor_mlp_state_dict)",
            "activation": "elu",
            "observation_normalization": False,
            "category_counts": list(M9B_CATEGORY_COUNTS),
        },
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = dict(scheduler_state_dict)
    return payload


def load_m11_bc_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11FactorizedActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_BC_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B checkpoint schema")
    raw = payload.get("model_config", {})
    config = M11BCModelConfig(
        observation_dim=int(raw["observation_dim"]),
        hidden_dims=tuple(int(value) for value in raw["hidden_dims"]),
        category_counts=tuple(int(value) for value in raw["category_counts"]),
        activation=str(raw.get("activation", "elu")),
        observation_normalization=bool(raw.get("observation_normalization", False)),
    )
    actor = M11FactorizedActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
