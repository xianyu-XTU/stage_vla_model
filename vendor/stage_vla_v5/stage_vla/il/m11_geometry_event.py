"""M11-B-D9 causal pre-action geometry CLOSE gate.

D8 proved that a learned scalar event score could not simultaneously suppress
all early runs and form a reliable positive window at the expert CLOSE token.
D9 therefore reuses the already-trained D8/D6 translation actor but removes the
learned CLOSE score from execution.  CLOSE is emitted only after the current
pre-action geometry has been safe for a configurable number of consecutive
steps.  Contact force remains a post-close grasp-verification signal and is not
used to request the first CLOSE action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from stage_vla.rl.action_dsl import (
    M9B_CLOSE_TOKEN,
    M9B_KEEP_TOKEN,
    M9B_OPEN_TOKEN,
)
from stage_vla.rl.m10_stage_core import M10_STAGE_COUNT
from stage_vla.stages.geometry import absolute_height_error, segment_projection

from .m11_bounded_event import (
    M11_D6_RESIDUAL_SCHEMA,
    M11_D8_CHECKPOINT_SCHEMA,
    M11BoundedEventActor,
    M11BoundedEventModelConfig,
)

M11_D9_CHECKPOINT_SCHEMA = "stage_vla.m11.geometry_ready_close_actor.v1"
M11_D9_SUMMARY_SCHEMA = "stage_vla.m11.geometry_ready_close_summary.v1"
M11_D9_CONTROLLER_SCHEMA = (
    "stage_vla.m11.pre_action_geometry_close_automatic_ready_open.v1"
)


@dataclass(frozen=True)
class M11PreActionGeometryDiagnostics:
    """Contact-free geometry used to decide whether the first CLOSE is safe."""

    geometry_ready: Tensor
    between_fingertips: Tensor
    left_height_aligned: Tensor
    right_height_aligned: Tensor
    projection_alpha: Tensor
    radial_error_m: Tensor
    left_height_error_m: Tensor
    right_height_error_m: Tensor


def m11_pre_action_geometry_diagnostics(
    red_pos_w: Tensor,
    left_tip_w: Tensor,
    right_tip_w: Tensor,
    *,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    endpoint_margin: float,
) -> M11PreActionGeometryDiagnostics:
    """Evaluate the M7-R8 expert's contact-free pre-CLOSE geometry predicate."""

    if not math.isfinite(radial_tolerance_m) or radial_tolerance_m < 0.0:
        raise ValueError("radial_tolerance_m must be finite and >= 0")
    if not math.isfinite(height_tolerance_m) or height_tolerance_m < 0.0:
        raise ValueError("height_tolerance_m must be finite and >= 0")
    if not math.isfinite(endpoint_margin) or not 0.0 <= endpoint_margin < 0.5:
        raise ValueError("endpoint_margin must satisfy 0 <= value < 0.5")

    projection = segment_projection(red_pos_w, left_tip_w, right_tip_w)
    inside = (
        (projection.alpha >= endpoint_margin)
        & (projection.alpha <= 1.0 - endpoint_margin)
    )
    between = inside & (projection.perpendicular_distance <= radial_tolerance_m)
    left_error = absolute_height_error(left_tip_w, red_pos_w)
    right_error = absolute_height_error(right_tip_w, red_pos_w)
    left_aligned = left_error <= height_tolerance_m
    right_aligned = right_error <= height_tolerance_m
    return M11PreActionGeometryDiagnostics(
        geometry_ready=between & left_aligned & right_aligned,
        between_fingertips=between,
        left_height_aligned=left_aligned,
        right_height_aligned=right_aligned,
        projection_alpha=projection.alpha,
        radial_error_m=projection.perpendicular_distance,
        left_height_error_m=left_error,
        right_height_error_m=right_error,
    )


class M11GeometryReadyController:
    """One-shot causal CLOSE from geometry, then deterministic ready-gated OPEN."""

    OPEN_STATE = 0
    CLOSED_STATE = 1
    RELEASED_STATE = 2

    def __init__(
        self,
        num_envs: int,
        *,
        allowed_close_stage_ids: Sequence[int],
        geometry_confirmation_steps: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        allowed = tuple(sorted({int(value) for value in allowed_close_stage_ids}))
        if not allowed or any(value < 0 or value >= M10_STAGE_COUNT for value in allowed):
            raise ValueError("allowed_close_stage_ids must contain valid M10 stages")
        if int(geometry_confirmation_steps) <= 0:
            raise ValueError("geometry_confirmation_steps must be > 0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.allowed_close_stage_ids = allowed
        self.geometry_confirmation_steps = int(geometry_confirmation_steps)
        self.state = torch.full(
            (self.num_envs,), self.OPEN_STATE, dtype=torch.long, device=self.device
        )
        self.geometry_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.maximum_geometry_run = torch.zeros_like(self.geometry_count)
        self.close_emissions = torch.zeros_like(self.geometry_count)
        self.open_emissions = torch.zeros_like(self.geometry_count)
        self.geometry_ready_frames = torch.zeros_like(self.geometry_count)
        self.unconfirmed_geometry_frames = torch.zeros_like(self.geometry_count)
        self.geometry_resets = torch.zeros_like(self.geometry_count)
        self.blocked_wrong_stage_geometry = torch.zeros_like(self.geometry_count)

    def reset(self, mask: Tensor | None = None) -> None:
        selected = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        )
        if selected.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [num_envs]")
        self.state[selected] = self.OPEN_STATE
        self.geometry_count[selected] = 0
        self.maximum_geometry_run[selected] = 0
        self.close_emissions[selected] = 0
        self.open_emissions[selected] = 0
        self.geometry_ready_frames[selected] = 0
        self.unconfirmed_geometry_frames[selected] = 0
        self.geometry_resets[selected] = 0
        self.blocked_wrong_stage_geometry[selected] = 0

    def step(
        self,
        stage_ids: Tensor,
        geometry_ready: Tensor,
        release_ready: Tensor,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        stages = torch.as_tensor(
            stage_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        geometry = torch.as_tensor(
            geometry_ready, dtype=torch.bool, device=self.device
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
        if any(value.shape != expected for value in (stages, geometry, ready, active)):
            raise ValueError("stage/geometry/ready/active must have shape [num_envs]")

        allowed_stage = torch.zeros_like(active)
        for stage_id in self.allowed_close_stage_ids:
            allowed_stage |= stages == int(stage_id)
        open_state = self.state == self.OPEN_STATE
        request = active & geometry
        qualifying = request & open_state & allowed_stage
        blocked = request & open_state & ~allowed_stage
        self.geometry_ready_frames += request.to(torch.long)
        self.blocked_wrong_stage_geometry += blocked.to(torch.long)

        reset_geometry = active & open_state & ~qualifying
        self.geometry_resets += (
            reset_geometry & (self.geometry_count > 0)
        ).to(torch.long)
        self.geometry_count[reset_geometry] = 0
        self.geometry_count[qualifying] += 1
        self.maximum_geometry_run = torch.maximum(
            self.maximum_geometry_run, self.geometry_count
        )
        close = qualifying & (
            self.geometry_count >= self.geometry_confirmation_steps
        )
        self.unconfirmed_geometry_frames += (qualifying & ~close).to(torch.long)

        # Capture the old state so same-frame geometry-ready + release-ready
        # cannot overwrite CLOSE with OPEN.
        was_closed = self.state == self.CLOSED_STATE
        self.state[close] = self.CLOSED_STATE
        self.geometry_count[close] = 0
        open_command = active & was_closed & ready

        tokens = torch.full(
            (self.num_envs,), M9B_KEEP_TOKEN, dtype=torch.long, device=self.device
        )
        tokens[close] = M9B_CLOSE_TOKEN
        tokens[open_command] = M9B_OPEN_TOKEN
        self.state[open_command] = self.RELEASED_STATE
        self.close_emissions += close.to(torch.long)
        self.open_emissions += open_command.to(torch.long)
        logits = torch.full(
            (self.num_envs, 3), -20.0, dtype=torch.float32, device=self.device
        )
        logits.scatter_(1, tokens.unsqueeze(-1), 20.0)
        return tokens, logits

    def summary(self) -> dict[str, Any]:
        return {
            "schema": M11_D9_CONTROLLER_SCHEMA,
            "environments": self.num_envs,
            "initial_state": "OPEN",
            "allowed_close_stage_ids": list(self.allowed_close_stage_ids),
            "close_trigger": "consecutive_pre_action_geometry_ready",
            "geometry_confirmation_steps": self.geometry_confirmation_steps,
            "learned_close_score_used": False,
            "contact_required_for_initial_close": False,
            "uses_post_action_truth_for_current_action": False,
            "automatic_open_on_pre_action_ready": True,
            "closed_state_environments": int(
                (self.state >= self.CLOSED_STATE).sum().item()
            ),
            "released_state_environments": int(
                (self.state == self.RELEASED_STATE).sum().item()
            ),
            "close_emissions": int(self.close_emissions.sum().item()),
            "open_emissions": int(self.open_emissions.sum().item()),
            "geometry_ready_frames": int(self.geometry_ready_frames.sum().item()),
            "unconfirmed_geometry_frames": int(
                self.unconfirmed_geometry_frames.sum().item()
            ),
            "geometry_resets": int(self.geometry_resets.sum().item()),
            "blocked_wrong_stage_geometry": int(
                self.blocked_wrong_stage_geometry.sum().item()
            ),
            "maximum_geometry_run_observed": int(
                self.maximum_geometry_run.max().item()
            ),
            "maximum_close_emissions_per_environment": int(
                self.close_emissions.max().item()
            ),
            "maximum_open_emissions_per_environment": int(
                self.open_emissions.max().item()
            ),
        }


def d9_checkpoint_payload(
    actor: M11BoundedEventActor,
    *,
    source_d8_payload: Mapping[str, Any],
    source_d8_checkpoint: str,
    allowed_close_stage_ids: Sequence[int],
    geometry_confirmation_steps: int,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    endpoint_margin: float,
    maximum_expert_close_lead_steps: int,
) -> dict[str, Any]:
    if source_d8_payload.get("schema") != M11_D8_CHECKPOINT_SCHEMA:
        raise ValueError("D9 source must use the loadable D8 checkpoint schema")
    if int(geometry_confirmation_steps) <= 0:
        raise ValueError("geometry_confirmation_steps must be > 0")
    if int(maximum_expert_close_lead_steps) < 0:
        raise ValueError("maximum_expert_close_lead_steps must be >= 0")
    if not math.isfinite(radial_tolerance_m) or radial_tolerance_m < 0.0:
        raise ValueError("radial_tolerance_m must be finite and >= 0")
    if not math.isfinite(height_tolerance_m) or height_tolerance_m < 0.0:
        raise ValueError("height_tolerance_m must be finite and >= 0")
    if not math.isfinite(endpoint_margin) or not 0.0 <= endpoint_margin < 0.5:
        raise ValueError("endpoint_margin must satisfy 0 <= value < 0.5")
    return {
        "schema": M11_D9_CHECKPOINT_SCHEMA,
        "model_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "source_d8_checkpoint": str(source_d8_checkpoint),
        "source_d8_schema": M11_D8_CHECKPOINT_SCHEMA,
        "source_d8_validation_metrics": source_d8_payload.get("validation_metrics"),
        "source_d8_validation_translation_exact_accuracy": source_d8_payload.get(
            "validation_translation_exact_accuracy"
        ),
        "source_d8_validation_residual_metrics": source_d8_payload.get(
            "validation_residual_metrics"
        ),
        "source_d8_close_metrics_diagnostic_only": source_d8_payload.get(
            "validation_close_event_metrics"
        ),
        "dataset": dict(source_d8_payload.get("dataset", {})),
        "residual_estimator_contract": dict(
            source_d8_payload.get("residual_estimator_contract", {})
        ),
        "controller_contract": {
            "schema": M11_D9_CONTROLLER_SCHEMA,
            "initial_state": "OPEN",
            "allowed_close_stage_ids": [
                int(value) for value in allowed_close_stage_ids
            ],
            "close_trigger": "consecutive_pre_action_geometry_ready",
            "geometry_confirmation_steps": int(geometry_confirmation_steps),
            "geometry_formula": (
                "between_fingertips AND left_height_aligned "
                "AND right_height_aligned"
            ),
            "radial_tolerance_m": float(radial_tolerance_m),
            "height_tolerance_m": float(height_tolerance_m),
            "endpoint_margin": float(endpoint_margin),
            "maximum_expert_close_lead_steps": int(
                maximum_expert_close_lead_steps
            ),
            "reads_current_pre_action_geometry": True,
            "learned_close_score_used": False,
            "contact_required_for_initial_close": False,
            "contact_reserved_for_post_close_grasp_verification": True,
            "uses_future_state": False,
            "uses_post_action_truth_for_current_action": False,
            "automatic_open_on_pre_action_ready": True,
            "learned_open_class_used": False,
            "maximum_close_emissions_per_episode": 1,
            "maximum_open_emissions_per_episode": 1,
        },
        "execution_contract": {
            "translation_actor_reused_without_retraining": True,
            "d8_event_head_ignored": True,
            "strict_success_definition_unchanged": True,
            "stage_reward_unchanged": True,
            "action_dsl_unchanged": True,
            "diagnostic_pilot_required_before_formal_strict": True,
            "ppo_authorized": False,
        },
        "ppo_authorized": False,
    }


def load_m11_d9_actor(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M11BoundedEventActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != M11_D9_CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported M11-B-D9 checkpoint schema")
    if payload.get("source_d8_schema") != M11_D8_CHECKPOINT_SCHEMA:
        raise RuntimeError("M11-B-D9 source D8 contract mismatch")

    residual = payload.get("residual_estimator_contract", {})
    if residual.get("schema") != M11_D6_RESIDUAL_SCHEMA:
        raise RuntimeError("M11-B-D9 residual estimator contract mismatch")
    if residual.get("causal_recurrent_history_only") is not True:
        raise RuntimeError("M11-B-D9 residual estimator must be causal")
    if residual.get("teacher_residual_is_training_target_only") is not True:
        raise RuntimeError("M11-B-D9 teacher residual must remain training-only")
    if residual.get("runtime_accumulator_used") is not False:
        raise RuntimeError("M11-B-D9 forbids a runtime residual accumulator")
    if residual.get("predicted_intent_accumulated") is not False:
        raise RuntimeError("M11-B-D9 forbids predicted-intent accumulation")

    controller = payload.get("controller_contract", {})
    if controller.get("schema") != M11_D9_CONTROLLER_SCHEMA:
        raise RuntimeError("M11-B-D9 geometry controller contract mismatch")
    if controller.get("close_trigger") != "consecutive_pre_action_geometry_ready":
        raise RuntimeError("M11-B-D9 CLOSE trigger contract mismatch")
    if controller.get("geometry_formula") != (
        "between_fingertips AND left_height_aligned AND right_height_aligned"
    ):
        raise RuntimeError("M11-B-D9 geometry formula contract mismatch")
    confirmation = controller.get("geometry_confirmation_steps")
    if not isinstance(confirmation, int) or confirmation <= 0:
        raise RuntimeError("M11-B-D9 geometry confirmation contract mismatch")
    if controller.get("reads_current_pre_action_geometry") is not True:
        raise RuntimeError("M11-B-D9 must read current pre-action geometry")
    if controller.get("learned_close_score_used") is not False:
        raise RuntimeError("M11-B-D9 must ignore the learned D8 CLOSE score")
    if controller.get("contact_required_for_initial_close") is not False:
        raise RuntimeError("M11-B-D9 cannot require contact before initial CLOSE")
    if controller.get("contact_reserved_for_post_close_grasp_verification") is not True:
        raise RuntimeError("M11-B-D9 contact verification contract mismatch")
    if controller.get("uses_future_state") is not False:
        raise RuntimeError("M11-B-D9 forbids future-state lookahead")
    if controller.get("uses_post_action_truth_for_current_action") is not False:
        raise RuntimeError("M11-B-D9 forbids post-action truth leakage")
    if controller.get("automatic_open_on_pre_action_ready") is not True:
        raise RuntimeError("M11-B-D9 automatic OPEN contract mismatch")
    if controller.get("learned_open_class_used") is not False:
        raise RuntimeError("M11-B-D9 must not restore learned OPEN")
    if controller.get("maximum_close_emissions_per_episode") != 1:
        raise RuntimeError("M11-B-D9 CLOSE one-shot contract mismatch")
    if controller.get("maximum_open_emissions_per_episode") != 1:
        raise RuntimeError("M11-B-D9 OPEN one-shot contract mismatch")
    allowed = controller.get("allowed_close_stage_ids")
    if (
        not isinstance(allowed, (list, tuple))
        or not allowed
        or any(
            not isinstance(value, int) or value < 0 or value >= M10_STAGE_COUNT
            for value in allowed
        )
    ):
        raise RuntimeError("M11-B-D9 CLOSE-stage contract mismatch")
    for name in ("radial_tolerance_m", "height_tolerance_m"):
        value = controller.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise RuntimeError(f"M11-B-D9 {name} contract mismatch")
    endpoint = controller.get("endpoint_margin")
    if (
        not isinstance(endpoint, (int, float))
        or not math.isfinite(float(endpoint))
        or not 0.0 <= float(endpoint) < 0.5
    ):
        raise RuntimeError("M11-B-D9 endpoint margin contract mismatch")
    maximum_lead = controller.get("maximum_expert_close_lead_steps")
    if not isinstance(maximum_lead, int) or maximum_lead < 0:
        raise RuntimeError("M11-B-D9 expert CLOSE lead contract mismatch")

    execution = payload.get("execution_contract", {})
    if execution.get("translation_actor_reused_without_retraining") is not True:
        raise RuntimeError("M11-B-D9 translation reuse contract mismatch")
    if execution.get("d8_event_head_ignored") is not True:
        raise RuntimeError("M11-B-D9 learned event head must be ignored")
    for name in (
        "strict_success_definition_unchanged",
        "stage_reward_unchanged",
        "action_dsl_unchanged",
        "diagnostic_pilot_required_before_formal_strict",
    ):
        if execution.get(name) is not True:
            raise RuntimeError(f"M11-B-D9 execution contract mismatch: {name}")
    if execution.get("ppo_authorized") is not False:
        raise RuntimeError("M11-B-D9 checkpoint cannot authorize PPO")
    if payload.get("ppo_authorized") is not False:
        raise RuntimeError("M11-B-D9 top-level checkpoint cannot authorize PPO")

    raw = payload.get("model_config", {})
    config = M11BoundedEventModelConfig(
        observation_dim=int(raw["observation_dim"]),
        observation_hidden_dims=tuple(
            int(value) for value in raw["observation_hidden_dims"]
        ),
        recurrent_hidden_dim=int(raw["recurrent_hidden_dim"]),
        head_hidden_dim=int(raw["head_hidden_dim"]),
        target_translation_scale_m=float(raw["target_translation_scale_m"]),
        category_counts=tuple(int(value) for value in raw["category_counts"]),
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
        raise RuntimeError("M11-B-D9 residual bound checkpoint mismatch")
    actor = M11BoundedEventActor(config)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(map_location)
    return actor, payload
