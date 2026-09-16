"""Portable analysis primitives for the M11-B-D1 closed-loop divergence audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from stage_vla.rl.action_dsl import M9B_CATEGORY_COUNTS
from stage_vla.rl.factorized_categorical_core import (
    deterministic_factor_tokens,
    split_factor_logits,
    validate_factor_tokens,
)

from .m11_behavior_cloning import M11_BC_FACTOR_NAMES, metrics_from_logits

M11_D1_REPORT_SCHEMA = "stage_vla.m11.bc_same_seed_divergence.v1"
M11_D1_SUMMARY_SCHEMA = "stage_vla.m11.bc_same_seed_divergence_summary.v1"


@dataclass(frozen=True)
class TokenDivergence:
    num_steps: int
    exact_action_accuracy: float
    exact_prefix_steps: int
    first_mismatch_step: int | None
    first_mismatch_factors: tuple[str, ...]
    factor_mismatch_counts: tuple[int, ...]
    factor_mismatch_rates: tuple[float, ...]


def token_divergence(
    predicted: Tensor,
    reference: Tensor,
    *,
    factor_names: Sequence[str] = M11_BC_FACTOR_NAMES,
) -> TokenDivergence:
    prediction = validate_factor_tokens(predicted, M9B_CATEGORY_COUNTS).cpu()
    target = validate_factor_tokens(reference, M9B_CATEGORY_COUNTS).cpu()
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("predicted/reference tokens must have equal [T,4] shape")
    if prediction.shape[0] == 0:
        raise ValueError("token divergence requires at least one step")
    if len(factor_names) != prediction.shape[1]:
        raise ValueError("factor_names length mismatch")

    mismatch = prediction != target
    any_mismatch = mismatch.any(dim=-1)
    mismatch_steps = torch.nonzero(any_mismatch, as_tuple=False).flatten()
    first = int(mismatch_steps[0].item()) if mismatch_steps.numel() else None
    factors = (
        tuple(
            str(factor_names[index])
            for index in torch.nonzero(mismatch[first], as_tuple=False).flatten().tolist()
        )
        if first is not None
        else tuple()
    )
    counts = mismatch.sum(dim=0).to(torch.long)
    num_steps = int(prediction.shape[0])
    return TokenDivergence(
        num_steps=num_steps,
        exact_action_accuracy=float((~any_mismatch).float().mean().item()),
        exact_prefix_steps=first if first is not None else num_steps,
        first_mismatch_step=first,
        first_mismatch_factors=factors,
        factor_mismatch_counts=tuple(int(value) for value in counts.tolist()),
        factor_mismatch_rates=tuple(float(value / num_steps) for value in counts.tolist()),
    )


def teacher_forced_audit(logits: Tensor, reference: Tensor) -> dict[str, Any]:
    target = validate_factor_tokens(reference, M9B_CATEGORY_COUNTS).to(logits.device)
    predicted = deterministic_factor_tokens(logits, M9B_CATEGORY_COUNTS)
    divergence = token_divergence(predicted, target)
    metrics = metrics_from_logits(logits, target)
    confidence = []
    for part, factor_target in zip(
        split_factor_logits(logits, M9B_CATEGORY_COUNTS), target.unbind(dim=-1), strict=True
    ):
        probabilities = torch.softmax(part, dim=-1)
        selected = probabilities.gather(1, factor_target.unsqueeze(-1)).squeeze(-1)
        confidence.append(float(selected.mean().item()))
    return {
        "metrics": asdict(metrics),
        "divergence": asdict(divergence),
        "mean_reference_token_probability": confidence,
    }


def translation_displacement_m(
    tokens: Tensor,
    *,
    full_scale_m: float = 0.004,
) -> Tensor:
    if full_scale_m <= 0.0:
        raise ValueError("full_scale_m must be > 0")
    value = validate_factor_tokens(tokens, M9B_CATEGORY_COUNTS)
    bins = value[..., :3].to(torch.float64) - 2.0
    return bins / 2.0 * float(full_scale_m)


def classify_d1_reports(
    train_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    expert_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify the next milestone without claiming more than the reports show."""
    for name, report in (("train", train_report), ("validation", validation_report)):
        if report.get("schema") != M11_D1_REPORT_SCHEMA:
            raise ValueError(f"{name} report schema mismatch")

    train_closed = train_report["closed_loop"]
    validation_closed = validation_report["closed_loop"]
    train_teacher = train_report["teacher_forced"]
    validation_teacher = validation_report["teacher_forced"]
    reset_tolerance = max(
        float(train_report["execution_contract"]["initial_observation_max_abs_error"]),
        float(validation_report["execution_contract"]["initial_observation_max_abs_error"]),
    )
    unit_scale = bool(train_report["execution_contract"]["unit_translation_scale"]) and bool(
        validation_report["execution_contract"]["unit_translation_scale"]
    )
    train_success = int(train_closed["strict_summary"]["strict_successes"])
    validation_success = int(validation_closed["strict_summary"]["strict_successes"])
    teacher_exact = min(
        float(train_teacher["metrics"]["exact_action_accuracy"]),
        float(validation_teacher["metrics"]["exact_action_accuracy"]),
    )

    evidence: list[str] = []
    if reset_tolerance > 1.0e-4:
        diagnosis = "RESET_OR_OBSERVATION_REPRODUCTION_MISMATCH"
        next_step = "fix same-seed reset/observation reproduction before changing the policy"
        evidence.append(f"initial observation max-abs error={reset_tolerance:.6g} > 1e-4")
    elif not unit_scale:
        diagnosis = "EXECUTION_SCALE_MISMATCH"
        next_step = "align collection and deployment translation scales before retraining"
        evidence.append("dataset/deployment translation scale ratio is not exactly one")
    elif (train_success > 0) != (validation_success > 0):
        diagnosis = "GENERALIZATION_OR_DATA_COVERAGE_GAP"
        next_step = "collect more accepted/perturbed layouts, then repeat validation-seed rollout"
        evidence.append(
            "accepted train/validation seeds have different closed-loop success outcomes"
        )
    elif train_success == 0 and validation_success == 0 and teacher_exact < 0.95:
        diagnosis = "SUPERVISED_JOINT_ACTION_ERROR_PLUS_CLOSED_LOOP_DRIFT"
        next_step = "run DAgger/noise-state aggregation before PPO; tighten exact-action diagnostics"
        evidence.append(f"minimum teacher-forced exact joint accuracy={teacher_exact:.4f} < 0.95")
    elif train_success == 0 and validation_success == 0:
        diagnosis = "CLOSED_LOOP_COVARIATE_SHIFT"
        next_step = "run DAgger with the R8 closed-loop expert before PPO"
        evidence.append("teacher forcing is strong but both accepted-seed free rollouts fail")
    else:
        diagnosis = "BC_CLOSES_ACCEPTED_SEEDS"
        next_step = "repeat multi-seed strict evaluation before deciding on M11-C PPO"
        evidence.append("BC achieves strict success on both accepted train and validation seeds")

    expert_check: dict[str, Any] | None = None
    if expert_summary is not None:
        if expert_summary.get("schema") != "stage_vla.m7_r8.randomized_acceptance.v1":
            raise ValueError("expert summary schema mismatch")
        successes = int(expert_summary["strict_successes"])
        total = int(expert_summary["num_seeds"])
        expert_check = {
            "strict_successes": successes,
            "num_seeds": total,
            "all_passed": successes == total,
        }
        if successes != total:
            evidence.append(
                f"R8 expert passes only {successes}/{total} on deployment seeds; evaluation coverage is not expert-verified"
            )

    return {
        "diagnosis": diagnosis,
        "next_step": next_step,
        "evidence": evidence,
        "train_strict_successes": train_success,
        "validation_strict_successes": validation_success,
        "minimum_teacher_forced_exact_action_accuracy": teacher_exact,
        "maximum_initial_observation_max_abs_error": reset_tolerance,
        "unit_translation_scale": unit_scale,
        "expert_seed_check": expert_check,
        "ppo_authorized": False,
    }
