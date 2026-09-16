"""M11-A-Q1 M7-R8 continuous-action traces -> factorized Action-DSL tokens.

The conversion is deliberately acceptance-gated.  A recorded M7-R8 command
uses the original Isaac task action scale, whereas F1 uses a 4 mm translation
scale.  Each source command is therefore decomposed into bounded F1-sized
subcommands and quantized to the existing five translation bins.  The result
is not considered an expert episode until it has been replayed from the same
seed and passes the unchanged strict M7 success observer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from stage_vla.rl.action_dsl import (
    M9B_CLOSE_TOKEN,
    M9B_KEEP_TOKEN,
    M9B_OPEN_TOKEN,
    M9B_TRANSLATION_CENTER_TOKEN,
    M9B_TRANSLATION_MAX_ABS_BIN,
)

M11_TRACE_SCHEMA = "stage_vla.m11.m7_continuous_trace.v1"
M11_DATASET_SCHEMA = "stage_vla.m11.dsl_demonstrations.v1"
M11_QUANTIZER_METHOD = "per_axis_error_feedback_v1"


@dataclass(frozen=True)
class M11QuantizationConfig:
    """Deterministic metric-aware conversion settings."""

    target_translation_scale_m: float = 0.004
    max_substeps_per_source_step: int = 32
    minimum_open_keep_steps: int = 8

    def __post_init__(self) -> None:
        if self.target_translation_scale_m <= 0.0:
            raise ValueError("target_translation_scale_m must be > 0")
        if self.max_substeps_per_source_step <= 0:
            raise ValueError("max_substeps_per_source_step must be > 0")
        if self.minimum_open_keep_steps <= 0:
            raise ValueError("minimum_open_keep_steps must be > 0")


@dataclass(frozen=True)
class M11TokenTrace:
    """Portable token sequence plus provenance needed for same-seed replay."""

    seed: int
    warmup_steps: int
    tokens: torch.Tensor
    phases: tuple[str, ...]
    source_step_indices: torch.Tensor
    source_action_scale_xyz: tuple[float, float, float]
    target_translation_scale_m: float
    final_translation_residual_m: tuple[float, float, float]
    max_abs_translation_residual_m: float


def _three_floats(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    out = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in out):
        raise ValueError(f"{name} contains NaN/Inf")
    return out


def validate_m7_trace_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != M11_TRACE_SCHEMA:
        raise ValueError(f"unsupported M11 trace schema: {payload.get('schema')!r}")
    if isinstance(payload.get("seed"), bool) or not isinstance(payload.get("seed"), int):
        raise TypeError("trace seed must be an integer")
    warmup = payload.get("warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("trace warmup_steps must be a non-negative integer")
    scale = _three_floats(payload.get("source_action_scale_xyz", ()), name="source_action_scale_xyz")
    if any(item <= 0.0 for item in scale):
        raise ValueError("source_action_scale_xyz values must be > 0")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("trace records must be a non-empty list")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"trace record {index} must be a mapping")
        if not isinstance(record.get("phase"), str) or not record["phase"]:
            raise ValueError(f"trace record {index} has no phase")
        action = record.get("raw_action")
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError(f"trace record {index} raw_action must contain seven values")
        if not all(math.isfinite(float(item)) for item in action):
            raise ValueError(f"trace record {index} raw_action contains NaN/Inf")


def _translation_token(raw_logical: torch.Tensor) -> torch.Tensor:
    """Nearest of {-1,-.5,0,.5,1}, returned as category indices 0..4."""
    clipped = raw_logical.clamp(-1.0, 1.0)
    bins = torch.round(clipped * 2.0).to(torch.long).clamp(-2, 2)
    return bins + M9B_TRANSLATION_CENTER_TOKEN


def _translation_token_metric_delta(
    token: torch.Tensor,
    *,
    target_translation_scale_m: float,
) -> torch.Tensor:
    """Decode XYZ categories to their exact metric replay displacement."""
    bins = token.to(torch.float64) - M9B_TRANSLATION_CENTER_TOKEN
    return (
        bins
        / float(M9B_TRANSLATION_MAX_ABS_BIN)
        * float(target_translation_scale_m)
    )


def _grip_token(desired_raw: float, previous_raw: float) -> tuple[int, float]:
    if not math.isfinite(desired_raw):
        raise ValueError("gripper command contains NaN/Inf")
    # M7 uses +/-1.  Zero is accepted only as KEEP for defensive portability.
    if desired_raw > 0.5:
        desired = 1.0
    elif desired_raw < -0.5:
        desired = -1.0
    else:
        desired = previous_raw
    if desired == previous_raw:
        return M9B_KEEP_TOKEN, previous_raw
    if desired > 0.0:
        return M9B_OPEN_TOKEN, 1.0
    return M9B_CLOSE_TOKEN, -1.0


def quantize_m7_trace(
    payload: Mapping[str, Any],
    *,
    cfg: M11QuantizationConfig = M11QuantizationConfig(),
) -> M11TokenTrace:
    """Convert one successful M7 trace to a bounded, stateful DSL sequence.

    Translation is converted in metric command space.  Large source commands
    are split so no target command exceeds the F1 4 mm envelope.  Q1 applies
    independent XYZ error-feedback accumulators before choosing the existing
    {-4,-2,0,+2,+4} mm replay displacement.  Quantization error therefore does
    not repeat on every shared substep: it is carried into the next token and
    remains bounded by half a bin (1 mm per axis at the default scale).  GRIP
    remains stateful: one transition token followed by KEEP, which yields
    exactly one terminal OPEN onset.
    """
    validate_m7_trace_payload(payload)
    source_scale = torch.tensor(
        _three_floats(payload["source_action_scale_xyz"], name="source_action_scale_xyz"),
        dtype=torch.float64,
    )
    tokens: list[torch.Tensor] = []
    phases: list[str] = []
    source_indices: list[int] = []
    previous_grip = 1.0
    residual_xyz_m = torch.zeros(3, dtype=torch.float64)
    max_abs_residual_m = 0.0
    residual_limit_m = (
        cfg.target_translation_scale_m
        / (2.0 * float(M9B_TRANSLATION_MAX_ABS_BIN))
    )

    for source_index, record in enumerate(payload["records"]):
        raw = torch.tensor(record["raw_action"], dtype=torch.float64)
        processed_xyz_m = raw[:3] * source_scale
        required = int(
            max(
                1,
                math.ceil(
                    float(processed_xyz_m.abs().max().item())
                    / cfg.target_translation_scale_m
                    - 1.0e-12
                ),
            )
        )
        if required > cfg.max_substeps_per_source_step:
            raise ValueError(
                f"source step {source_index} requires {required} substeps, exceeding "
                f"max_substeps_per_source_step={cfg.max_substeps_per_source_step}"
            )
        substeps = required
        ideal_substep_xyz_m = processed_xyz_m / float(substeps)
        first_grip_token, next_grip = _grip_token(float(raw[6].item()), previous_grip)
        for substep in range(substeps):
            # Each axis carries its own rounding error into the next token.
            # This prevents a non-dominant axis from repeating one rounded
            # displacement for every shared substep and drifting centimetres.
            requested_xyz_m = ideal_substep_xyz_m + residual_xyz_m
            xyz_token = _translation_token(
                requested_xyz_m / cfg.target_translation_scale_m
            )
            executed_xyz_m = _translation_token_metric_delta(
                xyz_token,
                target_translation_scale_m=cfg.target_translation_scale_m,
            )
            residual_xyz_m = requested_xyz_m - executed_xyz_m
            current_max_residual_m = float(residual_xyz_m.abs().max().item())
            max_abs_residual_m = max(max_abs_residual_m, current_max_residual_m)
            if current_max_residual_m > residual_limit_m + 1.0e-9:
                raise RuntimeError(
                    "M11-Q1 per-axis quantization residual exceeded half a bin: "
                    f"{current_max_residual_m:.9f} m > {residual_limit_m:.9f} m"
                )
            grip_token = first_grip_token if substep == 0 else M9B_KEEP_TOKEN
            tokens.append(
                torch.cat((xyz_token, torch.tensor([grip_token], dtype=torch.long)))
            )
            phases.append(str(record["phase"]))
            source_indices.append(source_index)
        previous_grip = next_grip

    token_tensor = torch.stack(tokens, dim=0)
    contract = terminal_grip_contract(
        token_tensor,
        minimum_open_keep_steps=cfg.minimum_open_keep_steps,
    )
    if not contract["valid"]:
        raise ValueError("quantized trace violates terminal GRIP contract: " + "; ".join(contract["errors"]))
    return M11TokenTrace(
        seed=int(payload["seed"]),
        warmup_steps=int(payload["warmup_steps"]),
        tokens=token_tensor,
        phases=tuple(phases),
        source_step_indices=torch.tensor(source_indices, dtype=torch.long),
        source_action_scale_xyz=tuple(float(item) for item in source_scale.tolist()),
        target_translation_scale_m=float(cfg.target_translation_scale_m),
        final_translation_residual_m=tuple(
            float(item) for item in residual_xyz_m.tolist()
        ),
        max_abs_translation_residual_m=float(max_abs_residual_m),
    )


def terminal_grip_contract(
    tokens: torch.Tensor,
    *,
    minimum_open_keep_steps: int = 8,
) -> dict[str, Any]:
    """Audit CLOSED-through-placement, one OPEN, then persistent KEEP-open."""
    value = torch.as_tensor(tokens)
    errors: list[str] = []
    if value.ndim != 2 or value.shape[1] != 4:
        return {"valid": False, "errors": ["tokens must have shape [T,4]"]}
    grip = value[:, 3].to(torch.long)
    close_indices = torch.nonzero(grip == M9B_CLOSE_TOKEN, as_tuple=False).flatten()
    open_indices = torch.nonzero(grip == M9B_OPEN_TOKEN, as_tuple=False).flatten()
    if close_indices.numel() != 1:
        errors.append(f"expected exactly one CLOSE transition, got {close_indices.numel()}")
    if open_indices.numel() != 1:
        errors.append(f"expected exactly one OPEN transition, got {open_indices.numel()}")
    open_index = int(open_indices[0].item()) if open_indices.numel() == 1 else -1
    close_index = int(close_indices[0].item()) if close_indices.numel() == 1 else -1
    if close_index >= 0 and open_index >= 0 and close_index >= open_index:
        errors.append("CLOSE must precede terminal OPEN")
    keep_after_open = 0
    if open_index >= 0:
        suffix = grip[open_index + 1 :]
        keep_after_open = int(suffix.numel())
        if not bool((suffix == M9B_KEEP_TOKEN).all().item()):
            errors.append("all GRIP tokens after terminal OPEN must be KEEP")
        if keep_after_open < minimum_open_keep_steps:
            errors.append(
                f"terminal OPEN requires at least {minimum_open_keep_steps} KEEP steps, got {keep_after_open}"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "close_index": close_index,
        "open_index": open_index,
        "keep_steps_after_open": keep_after_open,
    }


def trajectory_split(episode_id: int, *, validation_fraction: float = 0.2) -> str:
    """Stable trajectory-level 80/20 split without leaking steps across sets."""
    if episode_id < 0:
        raise ValueError("episode_id must be >= 0")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be inside (0,1)")
    period = max(2, int(round(1.0 / validation_fraction)))
    return "validation" if episode_id % period == period - 1 else "train"


def stack_bool_rows(rows: Iterable[Mapping[str, bool]], names: Sequence[str]) -> dict[str, torch.Tensor]:
    """Convert audited per-step truth dictionaries to compact bool tensors."""
    collected = list(rows)
    return {
        name: torch.tensor([bool(row[name]) for row in collected], dtype=torch.bool)
        for name in names
    }
