"""Persist whole-task evaluation results independently of video encoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .cli import EvaluationPlan


@dataclass(frozen=True)
class ResultContext:
    """Inputs needed to serialize one completed physical evaluation."""

    plan: "EvaluationPlan"
    prepared: Any
    v7_audit: Mapping[str, object]
    v7_chain_verified: bool
    runtime_purity: Mapping[str, object]
    provenance: Mapping[str, object]
    physical_success: bool
    reference_skill_calls: int
    recovery_calls: int
    passed: bool
    vision_stats: Mapping[str, object]
    vision_seed_valid: Any
    object_geometry: str
    grasp_profile: Any
    validation_envs: Sequence[int]
    stage_specs: Sequence[tuple[str, int, int]]
    relation_results: Sequence[dict[str, object]]
    overall_alive: Any
    reset_seen: bool
    reach_reference_recovery_used: bool
    demonstration_manifest: dict[str, object] | None
    dagger_manifest: dict[str, object] | None
    final_stack: dict[str, object] | None
    trace: list[dict[str, object]]
    recording_result: Any
    env: Any


def build_evaluation_result(context: ResultContext) -> dict[str, object]:
    """Assemble the stable evaluator JSON schema from completed subsystems."""
    plan = context.plan
    args = plan.args
    relation_results = context.relation_results
    primary = relation_results[0]
    single_relation = len(plan.task_pairs) == 1
    top_successes = (
        primary["successes"] if single_relation else {
            f"{item['asset_roles']['object']}->{item['asset_roles']['support']}":
                item["successes"]
            for item in relation_results
        }
    )
    result = {
        "status": "passed" if context.passed else "failed",
        "v7_chain": {
            "verified": context.v7_chain_verified,
            "required": bool(args.require_v7_chain),
            "command": plan.command_text,
            "planned_relations": [
                {
                    "object": relation.object_label,
                    "support": relation.support_label,
                }
                for relation in context.prepared.language.plan.execution_relations
            ],
            **context.v7_audit,
        },
        "runtime_purity": dict(context.runtime_purity),
        "provenance": dict(context.provenance),
        "closeout": {
            "physical_success": bool(context.physical_success),
            "stable_success": bool(
                context.relation_results
                and all(item.get("passed") is True for item in context.relation_results)
            ),
            "skills_exercised": sum(
                int(rows) > 0
                for rows in context.v7_audit.get("inference_rows_by_skill", {}).values()
            ),
            "skill_handoffs": sum(
                len(item.get("handoffs", ())) for item in context.relation_results
            ),
            "vision_service_calls": int(
                context.vision_stats.get("v7_service_calls", 0)
            ),
            "invalid_vision_frames": int(
                context.vision_stats.get("invalid_frames", 0)
            ),
            "oracle_fallback_count": int(
                context.vision_stats.get("oracle_fallback_count", 0)
            ),
            "reference_skill_calls": int(context.reference_skill_calls),
            "recovery_calls": int(context.recovery_calls),
            "runtime_purity_verified": context.runtime_purity.get("verified") is True,
            "v5_import_blocker_enabled": (
                context.runtime_purity.get("import_blocker_enabled") is True
            ),
            "v7_chain_verified": context.v7_chain_verified,
        },
        "scope": (
            "V7 language + V7 RGB-D service + V7 pipeline-routed TorchScript actions; "
            "oracle robot proprioception and physical terminal feedback; one initial "
            "reset, no inter-relation reset"
            if args.use_vision else
            "V7 language + V7 pipeline-routed TorchScript actions with oracle object "
            "positions; one initial reset, no inter-relation reset"
        ),
        "vision": {
            **context.vision_stats,
            "seed_valid": context.vision_seed_valid.tolist(),
            "backend": "compact_color_depth" if args.use_vision else None,
            "camera_calibration": (
                str(args.camera_calibration.resolve())
                if args.use_vision and args.camera_calibration is not None else None
            ),
            "geometry_bias_m": list(args.geometry_bias_m) if args.use_vision else None,
        },
        "episodes": args.num_envs,
        "scene_assets": list(plan.scene_assets),
        "validation_envs": list(context.validation_envs),
        "object_geometry": context.object_geometry,
        "asset_roles": primary["asset_roles"],
        "task_pairs": [
            {"object": object_asset, "support": support_asset}
            for object_asset, support_asset in plan.task_pairs
        ],
        "grasp_profile": {
            "geometry": context.grasp_profile.geometry,
            "height_ratio": context.grasp_profile.height_ratio,
            "width_ratio": context.grasp_profile.width_ratio,
        },
        "successes": top_successes,
        "chain_successes": (
            int(context.overall_alive[list(context.validation_envs)].sum())
        ),
        "seed_success": context.overall_alive.detach().cpu().tolist(),
        "mid_episode_resets": int(context.reset_seen),
        "reach_steps": (
            primary["reach_steps"] if single_relation
            else [item["reach_steps"] for item in relation_results]
        ),
        "reach_stable_steps": args.reach_stable_steps,
        "reach_recovery_steps": args.reach_recovery_steps,
        "reach_recovery_steps_used": (
            primary["reach_recovery_steps"] if single_relation else [
                item["reach_recovery_steps"] for item in relation_results
            ]
        ),
        "reach_recovery_steps_attempted": (
            primary["reach_recovery_steps_attempted"] if single_relation else [
                item["reach_recovery_steps_attempted"] for item in relation_results
            ]
        ),
        "reach_refine_steps": (
            primary["reach_refine_steps"] if single_relation else [
                item["reach_refine_steps"] for item in relation_results
            ]
        ),
        "reach_exit": (
            primary["reach_exit"] if single_relation
            else [item["reach_exit"] for item in relation_results]
        ),
        "checkpoints": {name: str(path) for name, path in plan.checkpoints.items()},
        "artifact_lock": (
            str(plan.artifact_lock) if plan.artifact_lock is not None else None
        ),
        "reach_checkpoint": (
            str(plan.reach_checkpoint) if plan.reach_checkpoint is not None else None
        ),
        "reach_controller": (
            args.reach_model_type
            if plan.reach_checkpoint is not None else "geometric_reference"
        ),
        "reach_refinement_controller": (
            args.reach_model_type
            if plan.reach_checkpoint is not None else "geometric_reference"
        ),
        "reach_recovery_controller": (
            "geometric_reference" if context.reach_reference_recovery_used else None
        ),
        "pure_policy_actions": bool(
            plan.reach_checkpoint is not None
            and not args.reference_skills
            and not context.reach_reference_recovery_used
        ),
        "reference_skills": list(args.reference_skills),
        "fixed_object_positions_local_xyz": plan.object_positions,
        "fixed_asset_positions_local_xyz": plan.fixed_asset_xyz,
        "asset_layout_file": (
            str(plan.asset_layout_file) if plan.asset_layout_file is not None else None
        ),
        "random_xy": bool(args.random_xy),
        "random_xy_seed": (
            args.seed if args.random_xy_seed is None else args.random_xy_seed
        ) if args.random_xy else None,
        "handoffs": primary["handoffs"] if single_relation else [
            item["handoffs"] for item in relation_results
        ],
        "interstage_conditioners": (
            primary["interstage_conditioners"] if single_relation else [
                item["interstage_conditioners"] for item in relation_results
            ]
        ),
        "horizons": {
            name: horizon for name, horizon, _stable in context.stage_specs
        },
        "translation_limits_m": plan.skill_translation_limits,
        "thresholds": {
            "align_xy_m": args.align_xy_m,
            "align_height_m": args.align_height_m,
            "align_height_tolerance_m": args.align_height_tolerance_m,
            "descend_height_m": args.descend_height_m,
            "descend_height_tolerance_m": args.descend_height_tolerance_m,
            "stack_xy_m": args.stack_xy_m,
            "stack_height_tolerance_m": args.stack_height_tolerance_m,
            "speed_mps": args.speed_mps,
            "physical_height_tolerance_m": args.physical_height_tolerance_m,
            "pregrasp_yaw_tolerance_rad": args.pregrasp_yaw_tolerance_rad,
            "angular_speed_source": context.env.stability_speed_source,
            "retreat_distance_m": args.retreat_distance_m,
            "retreat_height_m": args.retreat_height_m,
        },
        "demonstrations": context.demonstration_manifest,
        "dagger_labels": context.dagger_manifest,
        "rows": primary["rows"] if single_relation else None,
        "relations": list(relation_results),
        "final_stack": context.final_stack,
        "trace_env": args.trace_env,
        "trace": context.trace,
    }
    if args.video:
        if context.recording_result is None:
            raise RuntimeError("video recording result is missing")
        assert plan.observer_camera_model is not None
        result["video"] = {
            **context.recording_result.as_dict(),
            "env": int(args.video_env),
            "resolution": [int(args.video_width), int(args.video_height)],
            "camera": plan.observer_camera_model.name,
            "source": "observer-camera",
            "used_for_vision": False,
            "strict": bool(args.recording_strict),
        }
    return result


def write_json_result(path: Path, result: Mapping[str, object]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = ["ResultContext", "build_evaluation_result", "write_json_result"]
