"""Execute V7 Skill relations while preserving physical handoff state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .isolation import VisionIsolation
from .trace import PhysicalTraceCollector


@dataclass(frozen=True)
class ExecutionHelpers:
    """Frozen execution math helpers supplied by the V7 runtime."""

    skill_success: Callable[..., Any]
    jaw_leveling_axis_angle: Callable[..., Any]
    object_upright_tilt_rad: Callable[..., Any]
    grasp_target_position: Callable[..., Any]
    parallel_jaw_yaw_error: Callable[..., Any]
    project_pregrasp_edge_alignment: Callable[..., Any]
    reach_observation: Callable[..., Any]
    reach_raw_action: Callable[..., Any]
    pressure_tracking_ok: Callable[..., Any]
    hold_finished_skill_action: Callable[..., Any]


@dataclass(frozen=True)
class TaskExecutionContext:
    """Runtime collaborators needed to execute one planned relation chain."""

    args: Any
    raw: Any
    env: Any
    scene_assets: Sequence[str]
    task_pairs: Sequence[tuple[str, str]]
    grasp_profile: Any
    object_size_xyz: tuple[float, float, float]
    size: Any
    learned_reach: bool
    vision_seed_valid: Any
    output_module: Any
    reach_recovery_module: Any
    demonstration_dir: Path | None
    demonstration_rows: Any
    dagger_rows: Any
    validation_envs: Sequence[int]
    stage_specs: Sequence[tuple[str, int, int]]
    skill_translation_limits: Mapping[str, float]
    pipeline_action_source: Any
    measure_policy_reach_state: Callable[[str, str], Mapping[str, Any]]
    inject_visual_roles: Callable[[str, str], Any]
    capture_video_frame: Callable[[str, int], None]
    helpers: ExecutionHelpers


@dataclass(frozen=True)
class TaskExecutionResult:
    relation_results: list[dict[str, object]]
    reset_seen: bool
    overall_alive: Any
    trace: list[dict[str, object]]


def execute_task(context: TaskExecutionContext) -> TaskExecutionResult:
    """Run REACH and the seven downstream Skills for every planned relation."""
    import torch

    args = context.args
    raw = context.raw
    env = context.env
    scene_assets = context.scene_assets
    task_pairs = context.task_pairs
    grasp_profile = context.grasp_profile
    object_size_xyz = context.object_size_xyz
    size = context.size
    learned_reach = context.learned_reach
    vision_seed_valid = context.vision_seed_valid
    output_module = context.output_module
    reach_recovery_module = context.reach_recovery_module
    demonstration_dir = context.demonstration_dir
    demonstration_rows = context.demonstration_rows
    dagger_rows = context.dagger_rows
    validation_envs = context.validation_envs
    stage_specs = context.stage_specs
    skill_translation_limits = context.skill_translation_limits
    pipeline_action_source = context.pipeline_action_source
    measure_policy_reach_state = context.measure_policy_reach_state
    inject_visual_roles = context.inject_visual_roles
    capture_video_frame = context.capture_video_frame
    skill_success = context.helpers.skill_success
    grasp_target_position = context.helpers.grasp_target_position
    parallel_jaw_yaw_error = context.helpers.parallel_jaw_yaw_error
    project_pregrasp_edge_alignment = context.helpers.project_pregrasp_edge_alignment
    reach_observation = context.helpers.reach_observation
    reach_raw_action = context.helpers.reach_raw_action
    pressure_tracking_ok = context.helpers.pressure_tracking_ok
    hold_finished_skill_action = context.helpers.hold_finished_skill_action
    trace_collector = PhysicalTraceCollector(args, env, context.helpers)
    isolation = VisionIsolation(
        bool(args.use_vision), args.num_envs, vision_seed_valid, reach_raw_action
    )
    def scene_fingerprint():
        tensors = []
        for name in scene_assets:
            data = raw.unwrapped.scene[name].data
            tensors.append(torch.as_tensor(data.root_link_pose_w).clone())
            tensors.append(torch.as_tensor(data.root_com_vel_w).clone())
        tensors.append(torch.as_tensor(env.robot.data.joint_pos).clone())
        tensors.append(torch.as_tensor(env.robot.data.joint_vel).clone())
        return tuple(tensors)
    def run_reach(
        object_asset: str,
        support_asset: str,
        relation_index: int,
        active_mask,
    ):
        before_switch = scene_fingerprint()
        env.set_object_roles(object_asset, support_asset)
        after_switch = scene_fingerprint()
        role_switch_exact = all(
            torch.equal(before, after)
            for before, after in zip(before_switch, after_switch)
        )
        if not role_switch_exact:
            raise RuntimeError("physical state changed while switching object roles")

        label = f"relation {relation_index + 1} {object_asset}->{support_asset}"
        reach_state = measure_policy_reach_state(object_asset, support_asset)
        reach_state["grasp_target"] = grasp_target_position(
            reach_state["red"], object_size_xyz, profile=grasp_profile
        )
        reach_previous = torch.zeros(
            args.num_envs, 5, device=reach_state["ee"].device
        )
        active_mask = torch.as_tensor(
            active_mask, dtype=torch.bool, device=reach_state["ee"].device
        ).clone()
        active_mask &= isolation.mask(reach_state["ee"].device)
        reached = ~active_mask.clone()
        reach_stable = torch.zeros(
            args.num_envs, dtype=torch.long, device=reach_state["ee"].device
        )
        reach_steps_used = torch.zeros(
            args.num_envs, dtype=torch.long, device=reach_state["ee"].device
        )
        reach_reset = False
        recovery_steps_used = torch.zeros(
            args.num_envs, dtype=torch.long, device=reach_state["ee"].device
        )
        recovery_steps_attempted = 0
        label_reach = learned_reach and "REACH" in args.dagger_label_skills

        def project_reach_safety(action, active):
            if grasp_profile.geometry != "box":
                return action, torch.zeros_like(active), torch.zeros_like(
                    active, dtype=action.dtype
                )
            yaw_error = parallel_jaw_yaw_error(
                reach_state["red_quat"],
                env.object_size_m,
                reach_state["left_tip"],
                reach_state["right_tip"],
            )
            projected, aligning = project_pregrasp_edge_alignment(
                action,
                yaw_error,
                active,
                tolerance_rad=env.pregrasp_yaw_tolerance_rad,
            )
            return projected, aligning, yaw_error

        def reach_ready():
            valid = skill_success("REACH", reach_state)
            if grasp_profile.geometry == "box":
                yaw_error = parallel_jaw_yaw_error(
                    reach_state["red_quat"],
                    env.object_size_m,
                    reach_state["left_tip"],
                    reach_state["right_tip"],
                )
                valid &= yaw_error.abs() <= env.pregrasp_yaw_tolerance_rad
            return valid

        with torch.inference_mode():
            for step in range(1, args.reach_steps + 1):
                reach_obs = reach_observation(reach_state, reach_previous)
                output = output_module.emit(
                    "REACH",
                    reach_obs,
                    reference_state=reach_state,
                    finished=reached | ~active_mask,
                    inference_mask=active_mask,
                    translation_limit_m=0.005,
                    yaw_limit_rad=0.02,
                    include_reference=label_reach,
                )
                action = output.command
                active = output.active
                if demonstration_dir is not None and not learned_reach and bool(active.any()):
                    demonstration_rows.append("REACH", reach_obs[active], action[active])
                if label_reach and bool(active.any()):
                    if output.reference is None:
                        raise RuntimeError("missing DAgger reference action for REACH")
                    dagger_rows.append(
                        "REACH", reach_obs[active], output.reference[active]
                    )
                action, _aligning, _yaw_error = project_reach_safety(
                    action, active
                )
                raw.step(isolation.reach_raw_action(action, env.status.previous_action))
                reach_reset |= bool(raw.unwrapped.reset_buf.any())
                if reach_reset:
                    raise RuntimeError("simulator reset during REACH")
                capture_video_frame(f"{label} REACH", step)
                reach_state = measure_policy_reach_state(
                    object_asset, support_asset
                )
                active_mask &= isolation.mask(reach_state["ee"].device)
                reach_state["grasp_target"] = grasp_target_position(
                    reach_state["red"], object_size_xyz, profile=grasp_profile
                )
                valid = reach_ready()
                reach_stable = torch.where(
                    valid, reach_stable + 1, torch.zeros_like(reach_stable)
                )
                current_reached = (reach_stable >= args.reach_stable_steps) | ~active_mask
                newly = current_reached & ~reached
                reach_steps_used[newly] = step
                reached = current_reached
                reach_previous = action
                if bool(reached[list(validation_envs)].all()):
                    break
        if (
            not bool(reached[list(validation_envs)].all())
            and learned_reach
            and args.reach_recovery_steps
        ):
            for recovery_step in range(1, args.reach_recovery_steps + 1):
                recovery_steps_attempted = recovery_step
                reach_obs = reach_observation(reach_state, reach_previous)
                output = reach_recovery_module.emit(
                    "REACH",
                    reach_obs,
                    reference_state=reach_state,
                    finished=reached | ~active_mask,
                    inference_mask=active_mask,
                    translation_limit_m=0.005,
                    yaw_limit_rad=0.02,
                )
                action = output.command
                action, _aligning, _yaw_error = project_reach_safety(
                    action, output.active
                )
                raw.step(isolation.reach_raw_action(action, env.status.previous_action))
                reach_reset |= bool(raw.unwrapped.reset_buf.any())
                if reach_reset:
                    raise RuntimeError("simulator reset during REACH recovery")
                capture_video_frame(f"{label} REACH recovery", recovery_step)
                reach_state = measure_policy_reach_state(
                    object_asset, support_asset
                )
                active_mask &= isolation.mask(reach_state["ee"].device)
                reach_state["grasp_target"] = grasp_target_position(
                    reach_state["red"], object_size_xyz, profile=grasp_profile
                )
                valid = reach_ready()
                reach_stable = torch.where(
                    valid, reach_stable + 1, torch.zeros_like(reach_stable)
                )
                current_reached = (reach_stable >= args.reach_stable_steps) | ~active_mask
                newly = current_reached & ~reached
                recovery_steps_used[newly] = recovery_step
                reach_steps_used[newly] = args.reach_steps + recovery_step
                reached = current_reached
                reach_previous = action
                if bool(reached[list(validation_envs)].all()):
                    break
        if not bool((reached & active_mask).any()):
            missing = [
                index for index in validation_envs
                if bool(active_mask[index]) and not bool(reached[index])
            ]
            tip_mid = (reach_state["left_tip"] + reach_state["right_tip"]) / 2
            tip_error = tip_mid - reach_state["grasp_target"]
            attempted_steps = reach_steps_used.clone()
            attempted_steps[~reached] = args.reach_steps + args.reach_recovery_steps
            return env.observe(), {
                "passed": False,
                "env_success": (reached & active_mask).detach().cpu().tolist(),
                "steps": attempted_steps.detach().cpu().tolist(),
                "recovery_steps": recovery_steps_used.detach().cpu().tolist(),
                "recovery_steps_attempted": recovery_steps_attempted,
                "refine_steps": 0,
                "exit": {
                    "ee": reach_state["ee"].detach().cpu().tolist(),
                    "object": reach_state["red"].detach().cpu().tolist(),
                    "red": reach_state["red"].detach().cpu().tolist(),
                    "left_tip": reach_state["left_tip"].detach().cpu().tolist(),
                    "right_tip": reach_state["right_tip"].detach().cpu().tolist(),
                    "tip_target_error_m": tip_error.detach().cpu().tolist(),
                },
                "failure": {
                    "stage": "REACH",
                    "validation_envs": missing,
                    "final_tip_errors_m": tip_error[missing].detach().cpu().tolist(),
                },
                "reset_seen": reach_reset,
                "role_switch_physical_state_exact": role_switch_exact,
                "conditioners": [],
            }

        # Reference collection retains its historical fixed refinement window.
        refine_steps = args.reach_refine_steps if not learned_reach else 0
        with torch.inference_mode():
            for refine_step in range(1, refine_steps + 1):
                reach_obs = reach_observation(reach_state, reach_previous)
                output = output_module.emit(
                    "REACH",
                    reach_obs,
                    reference_state=reach_state,
                    finished=~active_mask,
                    inference_mask=active_mask,
                    translation_limit_m=0.005,
                    yaw_limit_rad=0.02,
                )
                action = output.command
                if demonstration_dir is not None and not learned_reach:
                    demonstration_rows.append("REACH", reach_obs, action)
                action, _aligning, _yaw_error = project_reach_safety(
                    action, output.active
                )
                raw.step(isolation.reach_raw_action(action, env.status.previous_action))
                if raw.unwrapped.reset_buf.any():
                    raise RuntimeError("simulator reset during REACH refinement")
                capture_video_frame(f"{label} REACH refine", refine_step)
                reach_state = measure_policy_reach_state(
                    object_asset, support_asset
                )
                active_mask &= isolation.mask(reach_state["ee"].device)
                reach_state["grasp_target"] = grasp_target_position(
                    reach_state["red"], object_size_xyz, profile=grasp_profile
                )
                reach_previous = action
        final_reach_valid = reach_ready()
        final_reach_success = final_reach_valid & reached & active_mask
        tip_mid = (reach_state["left_tip"] + reach_state["right_tip"]) / 2
        tip_error = tip_mid - reach_state["grasp_target"]
        reach_exit = {
            "ee": reach_state["ee"].detach().cpu().tolist(),
            "object": reach_state["red"].detach().cpu().tolist(),
            "red": reach_state["red"].detach().cpu().tolist(),
            "left_tip": reach_state["left_tip"].detach().cpu().tolist(),
            "right_tip": reach_state["right_tip"].detach().cpu().tolist(),
            "tip_target_error_m": tip_error.detach().cpu().tolist(),
        }
        if grasp_profile.geometry == "box":
            reach_exit["parallel_jaw_yaw_error_rad"] = (
                parallel_jaw_yaw_error(
                    reach_state["red_quat"],
                    env.object_size_m,
                    reach_state["left_tip"],
                    reach_state["right_tip"],
                ).detach().cpu().tolist()
            )
        conditioners = []
        if args.pregrasp_contact_descent_m:
            contact_action = torch.zeros(
                args.num_envs, 7, device=reach_state["ee"].device
            )
            contact_action[final_reach_success, 2] = -float(
                args.pregrasp_contact_descent_m
            ) / 0.005
            contact_action[:, 6] = 1.0
            raw.step(contact_action)
            if raw.unwrapped.reset_buf.any():
                raise RuntimeError("simulator reset during REACH->GRASP contact descent")
            capture_video_frame(f"{label} REACH->GRASP", 0)
            conditioners.append({
                "from": "REACH", "to": "GRASP",
                "type": "open_gripper_contact_micro_descent",
                "distance_m": args.pregrasp_contact_descent_m,
                "simulator_reset": False,
            })
        env.synchronize_after_external_step()
        inject_visual_roles(object_asset, support_asset)
        active_mask &= isolation.mask(reach_state["ee"].device)
        final_reach_success &= active_mask
        return env.observe(), {
            "passed": bool(final_reach_success.any()),
            "env_success": final_reach_success.detach().cpu().tolist(),
            "steps": reach_steps_used.detach().cpu().tolist(),
            "recovery_steps": recovery_steps_used.detach().cpu().tolist(),
            "recovery_steps_attempted": recovery_steps_attempted,
            "refine_steps": refine_steps,
            "exit": reach_exit,
            "reset_seen": reach_reset,
            "role_switch_physical_state_exact": role_switch_exact,
            "conditioners": conditioners,
        }

    def run_stage(obs, skill: str, horizon: int, relation_index: int, active_mask):
        rows: dict[int, dict] = {}
        first_action = None
        object_asset, support_asset = task_pairs[relation_index]
        active_mask = torch.as_tensor(
            active_mask, dtype=torch.bool, device=env.device
        ).clone()
        for step in range(1, horizon + 1):
            inject_visual_roles(object_asset, support_asset)
            active_mask &= isolation.mask(env.device)
            if not bool(active_mask.any()):
                break
            obs = env.observe()
            physical_state = env.physical_state
            status = env.status
            reference_state = dict(physical_state)
            reference_state["held"] = physical_state["physical"]
            reference_state["stack_height"] = torch.full(
                (args.num_envs,), args.descend_height_m, device=env.device
            )
            output = output_module.emit(
                skill,
                obs["policy"],
                reference_state=reference_state,
                finished=status.finished | ~active_mask,
                inference_mask=active_mask,
                translation_limit_m=skill_translation_limits.get(skill, 0.005),
                yaw_limit_rad=0.02,
                include_reference=skill in args.dagger_label_skills,
            )
            action = output.command
            if skill != "REACH":
                action = hold_finished_skill_action(
                    skill, action, status.finished | ~active_mask, status.previous_action
                )
            active = output.active
            if output.source == "reference":
                if demonstration_dir is not None and bool(active.any()):
                    demonstration_rows.append(
                        skill, obs["policy"][active], action[active]
                    )
            elif skill in args.dagger_label_skills and bool(active.any()):
                if output.reference is None:
                    raise RuntimeError(f"missing DAgger reference action for {skill}")
                dagger_rows.append(
                    skill, obs["policy"][active], output.reference[active]
                )
            if first_action is None:
                first_action = action.detach().cpu().clone()
            obs, _reward, done, _extras = env.step(action)
            record_done = done.clone()
            capture_video_frame(skill, step)
            trace_collector.append(
                relation_index=relation_index,
                skill=skill,
                step=step,
                checkpoint_action=output.command,
                submitted_action=action,
            )
            physical_state = env.physical_state
            status = env.status
            gripper = env.gripper_diagnostics
            for i in (record_done & active_mask).nonzero(as_tuple=False).flatten().tolist():
                rel = physical_state["red"][i] - physical_state["blue"][i]
                rows[i] = {
                    "env": i, "steps": step,
                    "success": bool(status.success[i]),
                    "failure": bool(status.failure[i]),
                    "timeout": bool(status.timeout[i]),
                    "physical_grasp": bool(physical_state["physical"][i]),
                    "pressure_ok": bool(pressure_tracking_ok(
                        physical_state["force"][i].unsqueeze(0),
                        gripper.target_force_n[i].unsqueeze(0), cfg=size
                    )[0]),
                    "force_n": physical_state["force"][i].detach().cpu().tolist(),
                    "target_force_n": float(gripper.target_force_n[i]),
                    "gripper_joint_m": physical_state["grip"][i].detach().cpu().tolist(),
                    "gripper_target_m": gripper.joint_target_m[i].detach().cpu().tolist(),
                    "fingertip_gap_m": float(physical_state["fingertip_gap_m"][i]),
                    "radial_error_m": float(physical_state["radial_error_m"][i]),
                    "left_height_error_m": float(physical_state["left_height_error_m"][i]),
                    "right_height_error_m": float(physical_state["right_height_error_m"][i]),
                    "stack_xy_m": float(rel[:2].norm()),
                    "stack_relative_height_m": float(rel[2]),
                    "lift_target_height_m": float(status.lift_target_height_m[i]),
                    "stability_speed_mps": float(physical_state["stability_speed"][i]),
                    "instantaneous_speed_mps": float(physical_state["speed"][i]),
                    "angular_speed_radps": float(
                        physical_state["stability_angular_speed"][i]
                    ),
                    "instantaneous_angular_speed_radps": float(
                        physical_state["instantaneous_angular_speed"][i]
                    ),
                    "control_angular_speed_radps": float(
                        physical_state["control_angular_speed"][i]
                    ),
                    "stable_steps": int(status.stable_count[i]),
                    "first_action": first_action[i].tolist(),
                    "terminal_action": action[i].detach().cpu().tolist(),
                }
            if bool(status.finished[active_mask].all()):
                break
        return obs, rows

    def handoff(obs, skill: str, horizon: int, stable_steps: int, active_mask):
        return env.configure_skill(
            skill,
            episode_steps=horizon,
            stable_steps=stable_steps,
            active_mask=active_mask,
        )

    def settle(obs, steps: int, label: str, relation_index: int, active_mask):
        if not steps:
            return obs
        action = torch.zeros(args.num_envs, 5, device=env.device)
        active = torch.as_tensor(active_mask, dtype=torch.bool, device=env.device)
        active &= isolation.mask(env.device)
        action[active, 4] = -1.0
        action[~active, 4] = env.status.previous_action[~active, 4]
        for step in range(1, steps + 1):
            obs, _reward, _done, _extras = env.step(action)
            if bool(env.unwrapped.reset_buf.any()):
                raise RuntimeError(f"simulator reset during {label} settle")
            capture_video_frame(label, step)
            trace_collector.append(
                relation_index=relation_index,
                skill=f"{env.current_skill}_SETTLE",
                step=step,
                submitted_action=action,
            )
        return obs
    relation_results = []
    reset_seen = False
    overall_alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)

    with torch.inference_mode():
        for relation_index, (object_asset, support_asset) in enumerate(task_pairs):
            pipeline_action_source.bind_relation(relation_index)
            relation_input = overall_alive.clone()
            print(
                f"[FULL CHAIN] starting relation {relation_index + 1} "
                f"{object_asset}->{support_asset}", flush=True,
            )
            obs, reach_result = run_reach(
                object_asset, support_asset, relation_index, relation_input
            )
            reset_seen |= reach_result["reset_seen"]
            active_mask = relation_input & torch.as_tensor(
                reach_result["env_success"], dtype=torch.bool, device=env.device
            )
            if not reach_result["passed"]:
                relation_results.append({
                    "asset_roles": {
                        "object": object_asset,
                        "support": support_asset,
                    },
                    "passed": False,
                    "env_success": active_mask.detach().cpu().tolist(),
                    "chain_successes": int(active_mask.sum()),
                    "failure": reach_result["failure"],
                    "successes": {},
                    "reach_steps": reach_result["steps"],
                    "reach_recovery_steps": reach_result["recovery_steps"],
                    "reach_recovery_steps_attempted": reach_result[
                        "recovery_steps_attempted"
                    ],
                    "reach_refine_steps": reach_result["refine_steps"],
                    "reach_exit": reach_result["exit"],
                    "role_switch_physical_state_exact": reach_result[
                        "role_switch_physical_state_exact"
                    ],
                    "handoffs": [],
                    "interstage_conditioners": [],
                    "rows": {},
                })
                print(
                    f"[FULL CHAIN] relation {relation_index + 1} REACH failed: "
                    f"{reach_result['failure']}",
                    flush=True,
                )
                break
            stage_rows: dict[str, dict[int, dict]] = {}
            successes: dict[str, int] = {}
            conditioners = list(reach_result["conditioners"])
            obs, exact = handoff(
                obs, "GRASP", args.grasp_steps, args.grasp_stable_steps,
                active_mask,
            )
            handoffs = [{
                "from": "REACH", "to": "GRASP",
                "physical_state_exact": exact,
            }]
            relation_label = f"relation {relation_index + 1} {object_asset}->{support_asset}"
            for index, (skill, horizon, stable_steps) in enumerate(stage_specs):
                if index:
                    previous = stage_specs[index - 1][0]
                    if not bool(active_mask.any()):
                        break
                    obs, exact = handoff(
                        obs, skill, horizon, stable_steps, active_mask
                    )
                    handoffs.append({
                        "from": previous, "to": skill,
                        "physical_state_exact": exact,
                    })
                print(f"[FULL CHAIN] {relation_label} running {skill}", flush=True)
                obs, stage_rows[skill] = run_stage(
                    obs, skill, horizon, relation_index, active_mask
                )
                reset_seen |= bool(env.unwrapped.reset_buf.any())
                active_mask &= isolation.mask(env.device)
                successes[skill] = sum(
                    row["success"] for row in stage_rows[skill].values()
                )
                stage_success = torch.zeros_like(active_mask)
                for env_index, row in stage_rows[skill].items():
                    stage_success[env_index] = bool(row["success"])
                active_mask &= stage_success
                print(
                    f"[FULL CHAIN] {relation_label} {skill}: "
                    f"{successes[skill]}/{args.num_envs}", flush=True,
                )
                if not bool(active_mask.any()):
                    break
                settle_steps = {
                    "GRASP": args.grasp_to_lift_settle_steps,
                    "LIFT": args.lift_to_transport_settle_steps,
                    "TRANSPORT": args.transport_to_align_settle_steps,
                    "ALIGN": args.align_to_descend_settle_steps,
                }.get(skill, 0)
                if settle_steps:
                    obs = settle(
                        obs,
                        settle_steps,
                        f"{relation_label} {skill} handoff",
                        relation_index,
                        active_mask,
                    )
                    conditioners.append({
                        "from": skill, "steps": settle_steps,
                        "type": "closed_gripper_physics_settle",
                        "simulator_reset": False,
                    })
            relation_complete = (
                len(successes) == len(stage_specs)
                and len(handoffs) == len(stage_specs)
                and all(item["physical_state_exact"] for item in handoffs)
                and reach_result["role_switch_physical_state_exact"]
            )
            if not relation_complete:
                active_mask.zero_()
            relation_passed = bool(
                relation_complete
                and active_mask[relation_input].all()
            )
            relation_results.append({
                "asset_roles": {"object": object_asset, "support": support_asset},
                "passed": relation_passed,
                "env_success": active_mask.detach().cpu().tolist(),
                "chain_successes": int(active_mask.sum()),
                "successes": successes,
                "reach_steps": reach_result["steps"],
                "reach_recovery_steps": reach_result["recovery_steps"],
                "reach_recovery_steps_attempted": reach_result[
                    "recovery_steps_attempted"
                ],
                "reach_refine_steps": reach_result["refine_steps"],
                "reach_exit": reach_result["exit"],
                "role_switch_physical_state_exact": reach_result[
                    "role_switch_physical_state_exact"
                ],
                "handoffs": handoffs,
                "interstage_conditioners": conditioners,
                "rows": {
                    skill: [rows[i] for i in sorted(rows)]
                    for skill, rows in stage_rows.items()
                },
            })
            overall_alive = active_mask
            if not bool(overall_alive.any()):
                break
    return TaskExecutionResult(
        relation_results=relation_results,
        reset_seen=reset_seen,
        overall_alive=overall_alive,
        trace=trace_collector.rows,
    )


__all__ = [
    "ExecutionHelpers",
    "TaskExecutionContext",
    "TaskExecutionResult",
    "execute_task",
]
