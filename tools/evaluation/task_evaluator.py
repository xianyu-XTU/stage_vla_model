"""Aggregate whole-task success without redefining per-Skill predicates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Whole-task aggregation produced after all requested relations run."""

    final_stack: dict[str, object] | None
    final_stack_successes: int
    required_final_stack_successes: int
    overall_alive: Any
    passed: bool


def evaluate_task(
    *,
    args: Any,
    raw: Any,
    env: Any,
    scene_assets: Sequence[str],
    task_pairs: Sequence[tuple[str, str]],
    relation_results: Sequence[dict[str, object]],
    reset_seen: bool,
    overall_alive: Any,
    validation_envs: Sequence[int],
    capture_video_frame: Callable[[str, int], None],
) -> TaskEvaluationResult:
    """Evaluate relation aggregation and the optional final multi-object stack."""
    import torch

    final_stack = None
    final_stack_successes = 0
    required_final_stack_successes = 0
    if len(task_pairs) > 1 and len(relation_results) == len(task_pairs):
        previous_stack_positions = {
            name: torch.as_tensor(
                raw.unwrapped.scene[name].data.root_pos_w
            )[..., :3].clone()
            for name in scene_assets
        }
        if args.final_stack_settle_steps:
            settle_action = torch.zeros(args.num_envs, 5, device=env.device)
            settle_action[:, 4] = 1.0
            for settle_step in range(1, args.final_stack_settle_steps + 1):
                previous_stack_positions = {
                    name: torch.as_tensor(
                        raw.unwrapped.scene[name].data.root_pos_w
                    )[..., :3].clone()
                    for name in scene_assets
                }
                env.step(settle_action)
                reset_seen |= bool(env.unwrapped.reset_buf.any())
                if reset_seen:
                    raise RuntimeError("simulator reset during final stack settle")
                capture_video_frame("final stack settle", settle_step)
        relation_checks = []
        all_valid = overall_alive.clone()
        for object_asset, support_asset in task_pairs:
            object_body = raw.unwrapped.scene[object_asset]
            support_body = raw.unwrapped.scene[support_asset]
            object_pos = torch.as_tensor(object_body.data.root_pos_w)[..., :3]
            support_pos = torch.as_tensor(support_body.data.root_pos_w)[..., :3]
            object_instantaneous_speed = torch.as_tensor(
                object_body.data.root_lin_vel_w
            )[..., :3].norm(dim=-1)
            support_instantaneous_speed = torch.as_tensor(
                support_body.data.root_lin_vel_w
            )[..., :3].norm(dim=-1)
            if args.final_stack_settle_steps:
                control_dt = max(float(raw.unwrapped.step_dt), 1e-6)
                object_speed = (
                    object_pos - previous_stack_positions[object_asset]
                ).norm(dim=-1) / control_dt
                support_speed = (
                    support_pos - previous_stack_positions[support_asset]
                ).norm(dim=-1) / control_dt
            else:
                object_speed = object_instantaneous_speed
                support_speed = support_instantaneous_speed
            relative = object_pos - support_pos
            valid = (
                (relative[:, :2].norm(dim=-1) <= args.stack_xy_m)
                & ((relative[:, 2] - args.descend_height_m).abs()
                   <= args.stack_height_tolerance_m)
                & (object_speed <= args.speed_mps)
                & (support_speed <= args.speed_mps)
            )
            all_valid &= valid
            relation_checks.append({
                "asset_roles": {"object": object_asset, "support": support_asset},
                "xy_m": relative[:, :2].norm(dim=-1).detach().cpu().tolist(),
                "relative_height_m": relative[:, 2].detach().cpu().tolist(),
                "object_speed_mps": object_speed.detach().cpu().tolist(),
                "support_speed_mps": support_speed.detach().cpu().tolist(),
                "object_instantaneous_speed_mps": (
                    object_instantaneous_speed.detach().cpu().tolist()
                ),
                "support_instantaneous_speed_mps": (
                    support_instantaneous_speed.detach().cpu().tolist()
                ),
                "valid": valid.detach().cpu().tolist(),
            })
        final_stack_successes = int(all_valid.sum())
        required_final_stack_successes = int(all_valid[list(validation_envs)].sum())
        overall_alive = all_valid
        final_stack = {
            "successes": final_stack_successes,
            "settle_steps": args.final_stack_settle_steps,
            "stability_speed_source": (
                "control_delta" if args.final_stack_settle_steps
                else "instantaneous"
            ),
            "relations": relation_checks,
        }

    passed = (
        len(relation_results) == len(task_pairs)
        and all(item["passed"] for item in relation_results)
        and not reset_seen
        and (
            len(task_pairs) == 1
            or required_final_stack_successes == len(validation_envs)
        )
    )
    return TaskEvaluationResult(
        final_stack=final_stack,
        final_stack_successes=final_stack_successes,
        required_final_stack_successes=required_final_stack_successes,
        overall_alive=overall_alive,
        passed=bool(passed),
    )


__all__ = ["TaskEvaluationResult", "evaluate_task"]
