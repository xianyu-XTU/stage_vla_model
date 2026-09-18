"""Run an audited V7 vision-language-action episode in Isaac Lab."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import numpy as np

from .bootstrap import V7_ROOT
from .audit import verify_v7_chain
from .camera_setup import build_evaluation_camera_specs, camera_cfg_transform
from .cli import EvaluationPlan
from .constants import ASSET_TO_VISION_LABEL, EXTRA_CUBE_COLORS
from .data_collection import SkillDemonstrationBuffer, write_data_manifests
from .debug_oracle import read_debug_oracle_local_positions
from .provenance import capture_source_snapshot, collect_evidence_provenance
from .result_writer import ResultContext, build_evaluation_result, write_json_result
from .runtime_purity import audit_runtime_purity, install_v5_import_blocker
from .task_evaluator import evaluate_task
from .task_executor import ExecutionHelpers, TaskExecutionContext, execute_task
from .vision_policy import VisionDetectionError, VisionFailPolicy, VisionPositionResolver
from stage_vla_v7.action import ActionOutputModule
from stage_vla_v7.action import (
    hold_finished_skill_action,
    jaw_leveling_axis_angle,
    object_upright_tilt_rad,
    project_pregrasp_edge_alignment,
)
from stage_vla_v7.contracts import ObjectDetection, SceneState, Skill
from stage_vla_v7.action.evaluation import SuccessChecker
from stage_vla_v7.orchestration import StageVLAPipeline, default_cube_catalog
from stage_vla_v7.simulation.config import load_known_size_config, load_object_metadata
from stage_vla_v7.simulation.environments import reach_observation
from stage_vla_v7.simulation.isaac_lab import (
    OBSERVER_CAMERA_NAME,
    VISION_CAMERA_NAME,
    IsaacCameraAdapter,
    PipelineActionSource,
    build_torchscript_cube_service,
    create_environment,
    measure_reach_state,
    reach_raw_action,
    to_torch,
)
from stage_vla_v7.simulation.recording import FrameCapture, RecordingConfig, VideoRecorder
from stage_vla_v7.simulation.physics import (
    grasp_target_position,
    local_width_ratio,
    parallel_jaw_yaw_error,
    pressure_tracking_ok,
    profile_from_config,
)
from stage_vla_v7.vision import (
    CameraCalibration,
    CompactColorDepthDetector,
    CompactColorDepthProvider,
    StaticVisionProvider,
    VisionRequest,
    VisionService,
)


def run_evaluation(plan: EvaluationPlan, app_launcher_class: Any) -> None:
    source_snapshot = capture_source_snapshot(V7_ROOT)
    args = plan.args
    scene_assets = plan.scene_assets
    skill_translation_limits = plan.skill_translation_limits
    task_pairs = plan.task_pairs
    command_text = plan.command_text
    language_service = plan.language_service
    label_to_asset = plan.label_to_asset
    fixed_asset_xyz = plan.fixed_asset_xyz
    asset_layout_file = plan.asset_layout_file
    checkpoints = plan.checkpoints
    reach_checkpoint = plan.reach_checkpoint
    artifact_lock = plan.artifact_lock
    expected_checkpoint_hashes = plan.expected_checkpoint_hashes
    out = plan.output_path
    demonstration_dir = plan.demonstration_dir
    dagger_dir = plan.dagger_dir
    video_path = plan.video_path
    object_positions = plan.object_positions
    observer_camera_model = plan.observer_camera_model

    try:
        install_v5_import_blocker()
    except RuntimeError as exc:
        write_json_result(out, {
            "status": "failed",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "v7_chain": {"verified": False, "required": bool(args.require_v7_chain)},
            "runtime_purity": audit_runtime_purity().as_dict(),
            "provenance": collect_evidence_provenance(
                plan,
                repository_root=V7_ROOT,
                source_snapshot=source_snapshot,
            ),
        })
        raise

    app = raw = env = None
    video_recorder = None
    recording_result = None
    vision_stats: dict[str, object] | None = None
    vision_seed_valid = None
    output_module = None
    reach_recovery_module = None
    try:
        app = app_launcher_class(args).app
        import torch
        from stage_vla_v7.simulation.isaac_lab.known_size_environment import (
            KnownSizeGraspEnvironment,
        )
        skill_success = SuccessChecker().evaluate_batch

        size = load_known_size_config(args.config.resolve())
        object_geometry, object_color_rgb = load_object_metadata(args.config.resolve())
        grasp_profile = profile_from_config(args.config.resolve(), geometry=object_geometry)
        size = replace(
            size,
            grasp_width_m=size.width_m * float(local_width_ratio(
                grasp_profile.geometry,
                grasp_profile.height_ratio,
                explicit_width_ratio=grasp_profile.width_ratio,
            ).item()),
        )
        camera_specs = build_evaluation_camera_specs(
            use_vision=bool(args.use_vision),
            observer=observer_camera_model if args.video else None,
            video_width=args.video_width,
            video_height=args.video_height,
            video_fps=args.video_fps,
        )
        raw = create_environment(
            "red_on_blue",
            device=args.device, num_envs=args.num_envs, seed=args.seed,
            log_dir=out.parent, known_size=size, effort_limit=args.effort_limit,
            camera_specs=camera_specs,
            camera_arm_scale=1.0,
            preserve_oracle_physics_with_camera=True,
            fixed_blue_xyz=object_positions["blue"] if object_positions is not None else None,
            fixed_red_xyz=object_positions["red"] if object_positions is not None else None,
            fixed_asset_xyz=fixed_asset_xyz,
            object_geometry=object_geometry,
            object_color_rgb=object_color_rgb,
            extra_cube_colors={
                name: EXTRA_CUBE_COLORS[int(name[5:]) - 4]
                for name in scene_assets[3:]
            },
        )
        camera_adapter = IsaacCameraAdapter()
        camera_bindings = camera_adapter.bind(
            raw.unwrapped.scene,
            use_vision=bool(args.use_vision),
            record_video=bool(args.video),
            vision_name=VISION_CAMERA_NAME,
            observer_name=(
                observer_camera_model.name
                if observer_camera_model is not None
                else OBSERVER_CAMERA_NAME
            ),
        )
        video_camera = camera_bindings.observer
        frame_capture = None
        if args.video:
            frame_capture = FrameCapture(args.video_width, args.video_height)
            video_recorder = VideoRecorder(RecordingConfig(
                path=video_path,
                width=args.video_width,
                height=args.video_height,
                fps=args.video_fps,
                strict=bool(args.recording_strict),
            ))
            video_recorder.start()

        def capture_video_frame(stage: str, step: int) -> None:
            if (
                video_camera is None
                or video_recorder is None
                or frame_capture is None
                or not video_recorder.active
            ):
                return
            try:
                payload = camera_adapter.observer_rgb_payload(
                    video_camera,
                    environment_index=args.video_env,
                )
                frame = frame_capture.capture(payload)
                video_recorder.capture(
                    frame=frame,
                    step=int(step),
                    skill=stage,
                    metadata={"seed": args.seed, "env_id": args.video_env},
                )
            except Exception as exc:
                video_recorder.fail(exc)
        # Construct the wrapper before REACH.  RslRlVecEnvWrapper performs an
        # initial reset in its constructor; constructing it after REACH would
        # silently destroy the physical REACH exit and invalidate the handoff.
        env = KnownSizeGraspEnvironment(
            raw, known_size=size, skill="GRASP", snapshots=(),
            episode_steps=args.grasp_steps, stable_steps=args.grasp_stable_steps,
            lift_translation_limit_m=args.lift_translation_limit_m,
            transport_planar_only=False, transport_xy_m=args.transport_xy_m,
            transport_translation_limit_m=args.transport_translation_limit_m,
            lift_vertical_only=True, align_xy_m=args.align_xy_m,
            align_height_target_m=args.align_height_m,
            align_height_tolerance_m=args.align_height_tolerance_m,
            align_speed_mps=args.speed_mps,
            align_translation_limit_m=args.align_translation_limit_m,
            descend_height_target_m=args.descend_height_m,
            descend_height_tolerance_m=args.descend_height_tolerance_m,
            descend_speed_mps=args.speed_mps,
            descend_translation_limit_m=args.descend_translation_limit_m,
            release_xy_m=args.stack_xy_m,
            release_height_tolerance_m=args.stack_height_tolerance_m,
            release_speed_mps=args.speed_mps,
            release_translation_limit_m=args.release_translation_limit_m,
            retreat_distance_m=args.retreat_distance_m,
            retreat_height_m=args.retreat_height_m,
            retreat_speed_mps=args.speed_mps,
            retreat_translation_limit_m=args.retreat_translation_limit_m,
            carry_leveling_max_angle_rad=args.carry_leveling_max_angle_rad,
            pregrasp_yaw_tolerance_rad=args.pregrasp_yaw_tolerance_rad,
            lift_arm_warmup_steps=args.lift_arm_warmup_steps,
            lift_arm_warmup_scale=args.lift_arm_warmup_scale,
            stability_speed_source="control_delta", arm_locked=True,
            seed=args.seed + 1, snapshot_assignment="sequential",
            grasp_profile=grasp_profile,
            object_asset_name=task_pairs[0][0],
            support_asset_name=task_pairs[0][1],
        )
        env.configure_evaluation(
            auto_reset=False,
            physical_height_tolerance_m=args.physical_height_tolerance_m,
        )

        vision_stats = {
            "enabled": bool(args.use_vision),
            "frames": 0,
            "v7_service_calls": 0,
            "fail_policy": args.vision_fail_policy,
            "strict_mode": bool(
                args.use_vision and args.vision_fail_policy == VisionFailPolicy.STRICT.value
            ),
            "invalid_frames": 0,
            "missing_by_asset": {name: 0 for name in scene_assets},
            "failed_objects": [],
            "oracle_fallback_used": False,
            "oracle_fallback_count": 0,
            "oracle_fallback_objects": [],
            "oracle_fallback_steps": [],
            "oracle_fallback_events": [],
        }
        vision_detector = None
        vision_camera = None
        vision_calibration = None
        vision_service = None
        vision_origins = None
        vision_seed_valid = np.ones(args.num_envs, dtype=bool)
        vision_position_resolver = None
        if args.use_vision:
            vision_position_resolver = VisionPositionResolver(
                VisionFailPolicy(args.vision_fail_policy),
                tuple(scene_assets),
                args.num_envs,
            )
            vision_camera = camera_bindings.vision
            assert vision_camera is not None
            vision_origins = to_torch(raw.unwrapped.scene.env_origins)[:, :3]
            intrinsic = to_torch(
                vision_camera.data.intrinsic_matrices
            )[0].detach().cpu().numpy()
            camera_position = to_torch(
                vision_camera.data.pos_w
            )[0].detach().cpu().numpy()
            camera_quat = to_torch(
                vision_camera.data.quat_w_ros
            )[0].detach().cpu().numpy()
            runtime_transform = camera_cfg_transform(
                camera_quat,
                camera_position,
                vision_origins[0].detach().cpu().numpy(),
            )
            if args.camera_calibration is not None:
                calibration_payload = json.loads(
                    args.camera_calibration.resolve().read_text(encoding="utf-8")
                )
                calibrated_intrinsic = np.asarray(
                    calibration_payload.get("intrinsic", intrinsic), dtype=np.float64
                )
                if not np.allclose(calibrated_intrinsic, intrinsic, atol=1e-3, rtol=1e-4):
                    raise RuntimeError(
                        "saved RGB-D calibration intrinsic does not match runtime camera"
                    )
                camera_to_root = np.asarray(
                    calibration_payload["camera_to_root"], dtype=np.float64
                )
            else:
                camera_to_root = runtime_transform
            vision_calibration = CameraCalibration(intrinsic, camera_to_root)
            vision_detector = CompactColorDepthDetector(
                min_pixels=args.vision_min_pixels,
                confidence_floor=args.vision_confidence_floor,
                position_bias_m=tuple(args.geometry_bias_m),
            )
            vision_service = VisionService(
                CompactColorDepthProvider(
                    detector=vision_detector,
                    calibration=vision_calibration,
                    labels=tuple(ASSET_TO_VISION_LABEL[name] for name in scene_assets),
                )
            )
            warmup_action = torch.zeros(
                (args.num_envs, 7), dtype=torch.float32, device=raw.unwrapped.device
            )
            for _ in range(3):
                raw.step(warmup_action)
                if bool(raw.unwrapped.reset_buf.any()):
                    raise RuntimeError("simulator reset during RGB-D warmup")
        else:
            origins = to_torch(raw.unwrapped.scene.env_origins)[:, :3]
            detections = []
            for asset_name in scene_assets:
                world = to_torch(raw.unwrapped.scene[asset_name].data.root_pos_w)[0, :3]
                local = (world - origins[0]).detach().cpu().tolist()
                detections.append(
                    ObjectDetection(ASSET_TO_VISION_LABEL[asset_name], tuple(local))
                )
            vision_service = VisionService(
                StaticVisionProvider(SceneState(tuple(detections), frame_id="isaac-oracle"))
            )

        def visual_asset_positions():
            """Read all cube positions from RGB-D, or return an empty mapping."""
            if not args.use_vision:
                return {}, np.ones(args.num_envs, dtype=bool)
            assert vision_camera is not None
            assert vision_detector is not None
            assert vision_calibration is not None
            assert vision_service is not None
            assert vision_origins is not None
            rgb = camera_adapter.rgb_u8_batch(
                vision_camera, camera_name=VISION_CAMERA_NAME
            )
            depth = camera_adapter.depth_m_batch(
                vision_camera, camera_name=VISION_CAMERA_NAME
            )
            predicted = {
                name: np.full((args.num_envs, 3), np.nan, dtype=np.float32)
                for name in scene_assets
            }
            for env_index in range(args.num_envs):
                observed = vision_service.observe(
                    VisionRequest(
                        rgb=rgb[env_index],
                        depth_m=depth[env_index],
                        frame_id=f"isaac-env-{env_index}",
                        metadata={"environment_index": env_index},
                    )
                )
                vision_stats["v7_service_calls"] += 1
                by_label = {item.label: item for item in observed.scene.detections}
                for asset_name in scene_assets:
                    detection = by_label.get(ASSET_TO_VISION_LABEL[asset_name])
                    if detection is not None:
                        predicted[asset_name][env_index] = np.asarray(
                            detection.position_xyz_m, dtype=np.float32
                        )
            vision_stats["frames"] += int(args.num_envs)
            assert vision_position_resolver is not None
            oracle_loader = None
            if args.vision_fail_policy == VisionFailPolicy.DEBUG_ORACLE.value:
                oracle_loader = lambda asset_name: read_debug_oracle_local_positions(
                    raw,
                    vision_origins,
                    asset_name,
                    to_torch=to_torch,
                )
            try:
                resolved, valid = vision_position_resolver.resolve(
                    predicted,
                    oracle_loader=oracle_loader,
                )
            finally:
                vision_stats.update(vision_position_resolver.audit())
            result = {}
            for asset_name, local_xyz in resolved.items():
                result[asset_name] = (
                    torch.as_tensor(local_xyz, device=env.device, dtype=torch.float32)
                    + vision_origins
                )
            vision_seed_valid[:] &= valid
            return result, valid

        def inject_visual_roles(object_asset: str, support_asset: str):
            if not args.use_vision:
                return np.ones(args.num_envs, dtype=bool)
            positions, valid = visual_asset_positions()
            env.set_visual_object_positions(
                positions[object_asset], positions[support_asset]
            )
            return valid

        def measure_policy_reach_state(object_asset: str, support_asset: str):
            state = measure_reach_state(
                raw.unwrapped,
                object_asset_name=object_asset,
                support_asset_name=support_asset,
            )
            if args.use_vision:
                positions, valid = visual_asset_positions()
                state["red"] = positions[object_asset]
                state["blue"] = positions[support_asset]
                env.set_visual_object_positions(state["red"], state["blue"])
            return state

        learned_reach = reach_checkpoint is not None
        object_size_xyz = (size.width_m, size.depth_m, size.height_m)
        demonstration_rows = SkillDemonstrationBuffer()
        dagger_rows = SkillDemonstrationBuffer(args.dagger_label_skills)
        if reach_checkpoint is None:
            action_checkpoints = {
                Skill.REACH: V7_ROOT / "tests" / "fixtures" / "unused-reference-reach.ts"
            }
        else:
            action_checkpoints = {Skill.REACH: reach_checkpoint}
        action_checkpoints.update({Skill(name): path for name, path in checkpoints.items()})
        action_service = (
            build_torchscript_cube_service(
                action_checkpoints,
                device=args.device,
                expected_hashes=expected_checkpoint_hashes,
            )
            if learned_reach
            else None
        )

        assert vision_service is not None
        if args.use_vision:
            initial_rgb = camera_adapter.rgb_u8_batch(
                vision_camera, camera_name=VISION_CAMERA_NAME
            )[0]
            initial_depth = camera_adapter.depth_m_batch(
                vision_camera, camera_name=VISION_CAMERA_NAME
            )[0]
            initial_frame = VisionRequest(
                rgb=initial_rgb,
                depth_m=initial_depth,
                frame_id="isaac-v7-prepare",
                metadata={"environment_index": 0},
            )
        else:
            initial_frame = VisionRequest(rgb=object(), frame_id="isaac-v7-oracle")
        if action_service is None:
            # Reference-only REACH collection has no complete learned V7 bundle.
            # It remains supported for data generation but is not a V7-chain proof.
            fallback_checkpoints = {Skill.REACH: checkpoints["GRASP"]}
            fallback_checkpoints.update({Skill(name): path for name, path in checkpoints.items()})
            action_service = build_torchscript_cube_service(
                fallback_checkpoints, device=args.device, bundle_name="v7-reference-data-cube"
            )
        pipeline = StageVLAPipeline(
            vision=vision_service,
            language=language_service,
            action=action_service,
            objects=default_cube_catalog(),
        )
        prepared = pipeline.prepare(command_text, initial_frame)
        planned_pairs = [
            (label_to_asset[relation.object_label], label_to_asset[relation.support_label])
            for relation in prepared.language.plan.execution_relations
        ]
        if planned_pairs != task_pairs:
            raise RuntimeError(
                f"V7 plan {planned_pairs!r} does not match simulator roles {task_pairs!r}"
            )
        pipeline_action_source = PipelineActionSource(pipeline, prepared)
        output_module = ActionOutputModule(
            pipeline_action_source,
            reference_skills=tuple(args.reference_skills) + (
                ("REACH",) if not learned_reach else ()
            ),
        )
        reach_recovery_module = ActionOutputModule(
            None, reference_skills=("REACH",)
        )
        stage_specs = (
            ("GRASP", args.grasp_steps, args.grasp_stable_steps),
            ("LIFT", args.lift_steps, args.lift_stable_steps),
            ("TRANSPORT", args.transport_steps, args.transport_stable_steps),
            ("ALIGN", args.align_steps, args.align_stable_steps),
            ("DESCEND", args.descend_steps, args.descend_stable_steps),
            ("RELEASE_STABILIZE", args.release_steps, args.release_stable_steps),
            ("RETREAT", args.retreat_steps, args.retreat_stable_steps),
        )
        validation_envs = (
            (args.video_env,) if args.video_env_only else tuple(range(args.num_envs))
        )
        execution = execute_task(TaskExecutionContext(
            args=args,
            raw=raw,
            env=env,
            scene_assets=scene_assets,
            task_pairs=task_pairs,
            grasp_profile=grasp_profile,
            object_size_xyz=object_size_xyz,
            size=size,
            learned_reach=learned_reach,
            vision_seed_valid=vision_seed_valid,
            output_module=output_module,
            reach_recovery_module=reach_recovery_module,
            demonstration_dir=demonstration_dir,
            demonstration_rows=demonstration_rows,
            dagger_rows=dagger_rows,
            validation_envs=validation_envs,
            stage_specs=stage_specs,
            skill_translation_limits=skill_translation_limits,
            pipeline_action_source=pipeline_action_source,
            measure_policy_reach_state=measure_policy_reach_state,
            inject_visual_roles=inject_visual_roles,
            capture_video_frame=capture_video_frame,
            helpers=ExecutionHelpers(
                skill_success=skill_success,
                jaw_leveling_axis_angle=jaw_leveling_axis_angle,
                object_upright_tilt_rad=object_upright_tilt_rad,
                grasp_target_position=grasp_target_position,
                parallel_jaw_yaw_error=parallel_jaw_yaw_error,
                project_pregrasp_edge_alignment=project_pregrasp_edge_alignment,
                reach_observation=reach_observation,
                reach_raw_action=reach_raw_action,
                pressure_tracking_ok=pressure_tracking_ok,
                hold_finished_skill_action=hold_finished_skill_action,
            ),
        ))
        relation_results = execution.relation_results
        reset_seen = execution.reset_seen
        overall_alive = execution.overall_alive

        task_evaluation = evaluate_task(
            args=args,
            raw=raw,
            env=env,
            scene_assets=scene_assets,
            task_pairs=task_pairs,
            relation_results=relation_results,
            reset_seen=reset_seen,
            overall_alive=overall_alive,
            validation_envs=validation_envs,
            capture_video_frame=capture_video_frame,
        )
        overall_alive = task_evaluation.overall_alive
        passed = task_evaluation.passed

        reach_reference_recovery_used = any(
            relation.get("reach_recovery_steps_attempted", 0) > 0
            for relation in relation_results
        )
        reference_skill_calls = output_module.reference_call_count
        recovery_calls = reach_recovery_module.reference_call_count
        v7_audit = pipeline_action_source.audit()
        runtime_purity = audit_runtime_purity().as_dict()
        v7_chain_verified = verify_v7_chain(
            use_vision=bool(args.use_vision),
            learned_reach=learned_reach,
            reference_skills=args.reference_skills,
            reach_reference_recovery_used=reach_reference_recovery_used,
            reference_skill_calls=reference_skill_calls,
            recovery_calls=recovery_calls,
            vision=vision_stats,
            pipeline_audit=v7_audit,
            runtime_purity=runtime_purity,
        )
        if args.require_v7_chain and not v7_chain_verified:
            passed = False

        manifests = write_data_manifests(
            demonstration_dir=demonstration_dir,
            dagger_dir=dagger_dir,
            demonstration_rows=demonstration_rows,
            dagger_rows=dagger_rows,
            passed=passed,
            object_geometry=object_geometry,
            grasp_profile=grasp_profile,
            output_path=out,
            reach_checkpoint=reach_checkpoint,
            checkpoints=checkpoints,
            reach_reference_recovery_used=reach_reference_recovery_used,
        )
        if args.video:
            assert video_recorder is not None
            recording_result = video_recorder.stop()

        result = build_evaluation_result(ResultContext(
            plan=plan,
            prepared=prepared,
            v7_audit=v7_audit,
            v7_chain_verified=v7_chain_verified,
            runtime_purity=runtime_purity,
            provenance=collect_evidence_provenance(
                plan,
                repository_root=V7_ROOT,
                source_snapshot=source_snapshot,
            ),
            physical_success=task_evaluation.passed,
            reference_skill_calls=reference_skill_calls,
            recovery_calls=recovery_calls,
            passed=passed,
            vision_stats=vision_stats,
            vision_seed_valid=vision_seed_valid,
            object_geometry=object_geometry,
            grasp_profile=grasp_profile,
            validation_envs=validation_envs,
            stage_specs=stage_specs,
            relation_results=relation_results,
            overall_alive=overall_alive,
            reset_seen=reset_seen,
            reach_reference_recovery_used=reach_reference_recovery_used,
            demonstration_manifest=manifests.demonstrations,
            dagger_manifest=manifests.dagger_labels,
            final_stack=task_evaluation.final_stack,
            trace=execution.trace,
            recording_result=recording_result,
            env=env,
        ))
        write_json_result(out, result)
        print(json.dumps(result, indent=2), flush=True)
        print(
            f"[V7 BENCHMARK] completed {result['chain_successes']}/{len(validation_envs)}",
            flush=True,
        )
    except Exception as exc:
        runtime_purity = audit_runtime_purity().as_dict()
        failure_result = {
            "status": "failed",
            "failure": (
                exc.as_dict() if isinstance(exc, VisionDetectionError)
                else {"type": type(exc).__name__, "message": str(exc)}
            ),
            "v7_chain": {
                "verified": False,
                "required": bool(args.require_v7_chain),
                "command": command_text,
            },
            "runtime_purity": runtime_purity,
            "provenance": collect_evidence_provenance(
                plan,
                repository_root=V7_ROOT,
                source_snapshot=source_snapshot,
            ),
            "reference_skill_calls": int(
                getattr(output_module, "reference_call_count", 0)
            ),
            "recovery_calls": int(
                getattr(reach_recovery_module, "reference_call_count", 0)
            ),
        }
        if vision_stats is not None:
            failure_result["vision"] = {
                **vision_stats,
                "seed_valid": (
                    vision_seed_valid.tolist() if vision_seed_valid is not None else None
                ),
            }
        write_json_result(out, failure_result)
        print(json.dumps(failure_result, indent=2), flush=True)
        # Isaac's application shutdown can terminate the process before the
        # interpreter prints an unhandled traceback.  Emit it while Kit is
        # still alive so failed physical evaluations remain diagnosable.
        import traceback

        traceback.print_exc()
        raise
    finally:
        if video_recorder is not None and video_recorder.active:
            try:
                video_recorder.stop()
            except Exception:
                pass
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        if app is not None:
            app.close()
