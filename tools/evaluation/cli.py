"""CLI parsing and fail-closed configuration for whole-task evaluation."""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

from .artifacts import load_action_checkpoint_hashes
from .bootstrap import V7_ROOT
from .constants import ASSET_TO_VISION_LABEL, EXTRA_CUBE_COLORS
from stage_vla_v7.contracts import Skill
from stage_vla_v7.language import DeterministicLanguageProvider, LanguageRequest, LanguageService
from stage_vla_v7.simulation.models import ObserverCameraModel
from stage_vla_v7.simulation.randomization import load_layout_manifest, sample_red_blue_batch


@dataclass(frozen=True)
class EvaluationPlan:
    args: argparse.Namespace
    scene_assets: tuple[str, ...]
    skill_translation_limits: dict[str, float]
    task_pairs: list[tuple[str, str]]
    command_text: str
    language_service: LanguageService
    label_to_asset: dict[str, str]
    fixed_asset_xyz: dict[str, object] | None
    asset_layout_file: Path | None
    checkpoints: dict[str, Path]
    reach_checkpoint: Path | None
    artifact_lock: Path | None
    expected_checkpoint_hashes: dict[Skill, str] | None
    output_path: Path
    demonstration_dir: Path | None
    dagger_dir: Path | None
    video_path: Path
    object_positions: dict[str, object] | None
    observer_camera_model: ObserverCameraModel | None


def scene_cube_assets(count: int) -> tuple[str, ...]:
    """Return stable Isaac asset names for a configurable cube scene."""
    if not 3 <= int(count) <= 3 + len(EXTRA_CUBE_COLORS):
        raise ValueError(f"scene cube count must be in [3,{3 + len(EXTRA_CUBE_COLORS)}]")
    return tuple(f"cube_{index}" for index in range(1, int(count) + 1))


def build_parser(app_launcher_class: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--reach_checkpoint",
        type=Path,
        default=None,
        help="optional independent REACH policy; omit for geometric reference",
    )
    parser.add_argument(
        "--reach_model_type",
        default="independent_bc_policy",
        help="controller type recorded when --reach_checkpoint is supplied",
    )
    for name in (
        "grasp",
        "lift",
        "transport",
        "align",
        "descend",
        "release",
        "retreat",
    ):
        parser.add_argument(f"--{name}_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact_lock",
        type=Path,
        default=None,
        help="optional V7 artifact lock that must cover and match all eight policies",
    )
    parser.add_argument(
        "--use_vision",
        action="store_true",
        help="replace object/support positions in policy observations with RGB-D detections",
    )
    parser.add_argument(
        "--require_v7_chain",
        action="store_true",
        help="fail the result unless vision, language, and every learned action use V7 services",
    )
    parser.add_argument("--camera_calibration", type=Path, default=None)
    parser.add_argument("--vision_confidence_floor", type=float, default=0.5)
    parser.add_argument("--vision_min_pixels", type=int, default=16)
    parser.add_argument("--geometry_bias_m", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--demonstration_dir",
        type=Path,
        default=None,
        help="save successful reference observations/actions for per-skill BC",
    )
    parser.add_argument(
        "--dagger_dir",
        type=Path,
        default=None,
        help="save teacher labels on states visited by model-controlled skills",
    )
    skill_names = (
        "REACH",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "ALIGN",
        "DESCEND",
        "RELEASE_STABILIZE",
        "RETREAT",
    )
    parser.add_argument(
        "--dagger_label_skills",
        nargs="*",
        default=(),
        choices=skill_names,
        help="model still acts; geometric teacher only labels visited states",
    )
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument(
        "--scene_cube_count",
        type=int,
        default=3,
        help="number of cube assets instantiated in the scene (3-7)",
    )
    parser.add_argument("--reach_steps", type=int, default=600)
    parser.add_argument(
        "--reach_stable_steps",
        type=int,
        default=3,
        help="consecutive geometry-valid steps required to finish REACH",
    )
    parser.add_argument(
        "--reach_refine_steps",
        type=int,
        default=12,
        help="open-gripper geometry corrections after all environments reach",
    )
    parser.add_argument(
        "--reach_recovery_steps",
        type=int,
        default=0,
        help="optional geometric REACH recovery after a learned REACH exhausts its horizon",
    )
    for name, default in (
        ("grasp", 80),
        ("lift", 100),
        ("transport", 180),
        ("align", 180),
        ("descend", 100),
        ("release", 180),
        ("retreat", 220),
    ):
        parser.add_argument(f"--{name}_steps", type=int, default=default)
    parser.add_argument(
        "--reference_skills",
        nargs="*",
        default=(),
        choices=skill_names[1:],
        help="diagnostic only: replace selected checkpoints with geometry reference actions",
    )
    for name, default in (
        ("grasp", 3),
        ("lift", 3),
        ("transport", 3),
        ("align", 3),
        ("descend", 3),
        ("release", 5),
        ("retreat", 20),
    ):
        parser.add_argument(f"--{name}_stable_steps", type=int, default=default)
    parser.add_argument("--grasp_to_lift_settle_steps", type=int, default=10)
    parser.add_argument(
        "--lift_translation_limit_m",
        type=float,
        default=None,
        help="maximum Cartesian LIFT translation per control step",
    )
    parser.add_argument(
        "--pregrasp_contact_descent_m",
        type=float,
        default=0.0,
        help="deterministic REACH->GRASP contact-depth conditioner",
    )
    parser.add_argument("--lift_to_transport_settle_steps", type=int, default=10)
    parser.add_argument("--transport_to_align_settle_steps", type=int, default=10)
    parser.add_argument("--align_to_descend_settle_steps", type=int, default=10)
    parser.add_argument(
        "--final_stack_settle_steps",
        type=int,
        default=0,
        help="open-gripper physics-only settling before the final multi-object stack check",
    )
    parser.add_argument("--transport_xy_m", type=float, default=0.045)
    for name in ("transport", "align", "descend", "release", "retreat"):
        parser.add_argument(f"--{name}_translation_limit_m", type=float, default=None)
    parser.add_argument("--align_xy_m", type=float, default=0.010)
    parser.add_argument("--align_height_m", type=float, default=0.0618)
    parser.add_argument("--align_height_tolerance_m", type=float, default=0.015)
    parser.add_argument("--descend_height_m", type=float, default=0.0468)
    parser.add_argument("--descend_height_tolerance_m", type=float, default=0.002)
    parser.add_argument("--stack_xy_m", type=float, default=0.040)
    parser.add_argument("--stack_height_tolerance_m", type=float, default=0.010)
    parser.add_argument("--speed_mps", type=float, default=0.05)
    parser.add_argument("--physical_height_tolerance_m", type=float, default=0.015)
    parser.add_argument("--retreat_distance_m", type=float, default=0.100)
    parser.add_argument("--retreat_height_m", type=float, default=0.080)
    parser.add_argument("--effort_limit", type=float, default=40.0)
    parser.add_argument(
        "--carry_leveling_max_angle_rad",
        type=float,
        default=0.0,
        help="bounded roll/pitch correction that keeps the parallel-jaw axis level",
    )
    parser.add_argument(
        "--pregrasp_yaw_tolerance_rad",
        type=float,
        default=0.005,
        help="maximum box-edge yaw error before GRASP is allowed to close",
    )
    parser.add_argument("--lift_arm_warmup_steps", type=int, default=0)
    parser.add_argument("--lift_arm_warmup_scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument(
        "--object_asset",
        default="cube_2",
        help="Isaac scene asset assigned to the manipulated-object role",
    )
    parser.add_argument(
        "--support_asset",
        default="cube_1",
        help="Isaac scene asset assigned to the support role",
    )
    parser.add_argument(
        "--task_pairs",
        nargs="+",
        default=None,
        metavar="OBJECT:SUPPORT",
        help=(
            "ordered object/support relations executed in one simulator episode; "
            "for example cube_1:cube_3 cube_2:cube_1"
        ),
    )
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "natural-language or DSL stack command; when supplied it owns task "
            "planning and must resolve to assets in the configured scene"
        ),
    )
    parser.add_argument(
        "--blue_xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="fixed blue cube position in the local environment frame",
    )
    parser.add_argument(
        "--red_xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="fixed red cube position in the local environment frame",
    )
    parser.add_argument(
        "--asset_xyz",
        nargs=4,
        action="append",
        default=None,
        metavar=("ASSET", "X", "Y", "Z"),
        help="fixed local position for one scene asset; repeat for multi-object layouts",
    )
    parser.add_argument(
        "--asset_layout_file",
        type=Path,
        default=None,
        help="JSON layout batch with one XYZ row per asset and parallel environment",
    )
    parser.add_argument(
        "--random_xy",
        action="store_true",
        help="sample one blue/red XY pair per environment; Z remains 0.0203 m",
    )
    parser.add_argument(
        "--random_xy_seed",
        type=int,
        default=None,
        help="seed for reproducible XY pair generation (defaults to --seed)",
    )
    parser.add_argument(
        "--video", action="store_true", help="record one environment's RGB trajectory to an MP4"
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="output MP4 path; defaults to output JSON path with .mp4",
    )
    parser.add_argument(
        "--video_env",
        type=int,
        default=0,
        help="environment index to record when --video is enabled",
    )
    parser.add_argument(
        "--video_env_only",
        action="store_true",
        help="gate a recording run on the recorded environment instead of every parallel environment",
    )
    parser.add_argument("--video_width", type=int, default=None)
    parser.add_argument("--video_height", type=int, default=None)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument(
        "--observer_camera_config",
        type=Path,
        default=V7_ROOT / "config" / "simulation" / "observer_camera.json",
        help="third-person camera configuration; never used as Vision input",
    )
    recording_mode = parser.add_mutually_exclusive_group()
    recording_mode.add_argument(
        "--recording_strict",
        dest="recording_strict",
        action="store_true",
        help="fail the episode when observer recording fails (default)",
    )
    recording_mode.add_argument(
        "--recording_best_effort",
        dest="recording_strict",
        action="store_false",
        help="warn and continue control when observer recording fails",
    )
    parser.set_defaults(recording_strict=True)
    parser.add_argument(
        "--trace_env",
        type=int,
        default=None,
        help="optional environment index for per-step physical diagnostics",
    )
    app_launcher_class.add_app_launcher_args(parser)
    return parser


def _load_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command is not None:
        return
    encoded_command = os.environ.get("STAGE_VLA_COMMAND_UTF8_BASE64")
    if encoded_command:
        try:
            args.command = base64.b64decode(encoded_command, validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
            parser.error(f"invalid STAGE_VLA_COMMAND_UTF8_BASE64: {exc}")
    else:
        args.command = os.environ.get("STAGE_VLA_COMMAND") or None


def _validate_numeric_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, float]:
    from stage_vla.stages.object_motion import motion_profile_from_config

    if not 0.0 < args.vision_confidence_floor <= 1.0:
        parser.error("--vision_confidence_floor must be in (0,1]")
    if args.vision_min_pixels < 1:
        parser.error("--vision_min_pixels must be positive")
    motion_profile = motion_profile_from_config(args.config.resolve())
    for field in motion_profile.__dataclass_fields__:
        if getattr(args, field) is None:
            setattr(args, field, getattr(motion_profile, field))
    skill_translation_limits = {
        "LIFT": args.lift_translation_limit_m,
        "TRANSPORT": args.transport_translation_limit_m,
        "ALIGN": args.align_translation_limit_m,
        "DESCEND": args.descend_translation_limit_m,
        "RELEASE_STABILIZE": args.release_translation_limit_m,
        "RETREAT": args.retreat_translation_limit_m,
    }
    positive = (
        args.num_envs,
        args.reach_steps,
        args.reach_stable_steps,
        args.grasp_steps,
        args.lift_steps,
        args.transport_steps,
        args.align_steps,
        args.descend_steps,
        args.release_steps,
        args.retreat_steps,
        args.grasp_stable_steps,
        args.lift_stable_steps,
        args.transport_stable_steps,
        args.align_stable_steps,
        args.descend_stable_steps,
        args.release_stable_steps,
        args.retreat_stable_steps,
    )
    if min(positive) < 1:
        parser.error("counts, horizons and stable steps must be positive")
    settles = (
        args.grasp_to_lift_settle_steps,
        args.lift_to_transport_settle_steps,
        args.transport_to_align_settle_steps,
        args.align_to_descend_settle_steps,
        args.final_stack_settle_steps,
    )
    if min(*settles, args.reach_recovery_steps) < 0:
        parser.error("settle and recovery steps must be non-negative")
    if not 0.0 <= args.pregrasp_contact_descent_m <= 0.005:
        parser.error("pregrasp_contact_descent_m must be in [0, 0.005]")
    motion_limits = tuple(
        getattr(args, field) for field in motion_profile.__dataclass_fields__
    )
    if min(args.physical_height_tolerance_m, *motion_limits) <= 0:
        parser.error("physical tolerances and translation limits must be positive")
    return skill_translation_limits


def _load_observer(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> ObserverCameraModel | None:
    if args.video and not 0 <= args.video_env < args.num_envs:
        parser.error("video_env must be a valid environment index")
    if args.video_env_only and not args.video:
        parser.error("--video_env_only requires --video")
    if not args.video:
        return None
    observer_config = args.observer_camera_config.resolve()
    if not observer_config.is_file():
        parser.error(f"observer camera config does not exist: {observer_config}")
    try:
        observer = ObserverCameraModel.from_json(observer_config)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid observer camera config: {exc}")
    if args.video_width is None:
        args.video_width = observer.width
    if args.video_height is None:
        args.video_height = observer.height
    if args.video_fps is None:
        args.video_fps = observer.fps
    if min(args.video_width, args.video_height) < 1:
        parser.error("video dimensions must be positive")
    if args.video_fps <= 0:
        parser.error("video_fps must be positive")
    return observer


def _resolve_task(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    scene_assets: tuple[str, ...],
) -> tuple[list[tuple[str, str]], str, LanguageService, dict[str, str]]:
    if args.trace_env is not None and not 0 <= args.trace_env < args.num_envs:
        parser.error("trace_env must be in [0, num_envs)")
    if (args.blue_xyz is None) != (args.red_xyz is None):
        parser.error("--blue_xyz and --red_xyz must be supplied together")
    if args.random_xy and args.blue_xyz is not None:
        parser.error("--random_xy cannot be combined with --blue_xyz/--red_xyz")
    if args.asset_xyz is not None and (args.random_xy or args.blue_xyz is not None):
        parser.error("--asset_xyz cannot be combined with pair position overrides")
    if args.asset_layout_file is not None and (
        args.asset_xyz is not None or args.random_xy or args.blue_xyz is not None
    ):
        parser.error("--asset_layout_file cannot be combined with other position overrides")
    if args.object_asset == args.support_asset:
        parser.error("--object_asset and --support_asset must differ")
    if args.object_asset not in scene_assets or args.support_asset not in scene_assets:
        parser.error("--object_asset and --support_asset must exist in the configured cube scene")
    task_pairs = [(args.object_asset, args.support_asset)]
    if args.task_pairs is not None:
        task_pairs = []
        valid_assets = set(scene_assets)
        for value in args.task_pairs:
            parts = value.split(":")
            if len(parts) != 2 or parts[0] not in valid_assets or parts[1] not in valid_assets:
                parser.error(f"invalid --task_pairs value {value!r}; expected cube_N:cube_N")
            if parts[0] == parts[1]:
                parser.error(f"task pair roles must differ: {value!r}")
            task_pairs.append((parts[0], parts[1]))
    language_service = LanguageService(DeterministicLanguageProvider())
    label_to_asset = {
        label: asset
        for asset, label in ASSET_TO_VISION_LABEL.items()
        if asset in scene_assets
    }
    if args.command is not None:
        try:
            language_result = language_service.interpret(
                LanguageRequest(args.command, tuple(label_to_asset))
            )
            command_pairs = [
                (label_to_asset[relation.object_label], label_to_asset[relation.support_label])
                for relation in language_result.plan.execution_relations
            ]
        except (KeyError, ValueError, LookupError) as exc:
            parser.error(f"invalid --command: {exc}")
        if args.task_pairs is not None and command_pairs != task_pairs:
            parser.error("--command and --task_pairs resolve to different execution plans")
        task_pairs = command_pairs
        command_text = args.command
    else:
        clauses = [
            f"{ASSET_TO_VISION_LABEL[obj]}>{ASSET_TO_VISION_LABEL[support]}"
            for obj, support in task_pairs
        ]
        command_text = (
            clauses[0] if len(clauses) == 1 else f"STACK_CHAIN({','.join(clauses)})"
        )
    if (args.object_asset, args.support_asset) != ("cube_2", "cube_1") and (
        args.random_xy or args.blue_xyz is not None
    ):
        parser.error("fixed/random pair overrides currently require cube_2 -> cube_1 roles")
    if len(task_pairs) > 1 and (args.random_xy or args.blue_xyz is not None):
        parser.error("multi-relation evaluation cannot use fixed/random pair overrides")
    return task_pairs, command_text, language_service, label_to_asset


def _resolve_layout(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    scene_assets: tuple[str, ...],
) -> tuple[dict[str, object] | None, Path | None, dict[str, object] | None]:
    fixed_asset_xyz: dict[str, object] | None = None
    if args.asset_xyz is not None:
        fixed_asset_xyz = {}
        for name, *coordinates in args.asset_xyz:
            if name not in set(scene_assets):
                parser.error(f"invalid asset name in --asset_xyz: {name!r}")
            if name in fixed_asset_xyz:
                parser.error(f"duplicate --asset_xyz entry: {name!r}")
            try:
                xyz = tuple(float(value) for value in coordinates)
            except ValueError:
                parser.error(f"invalid coordinates for --asset_xyz {name!r}")
            if not all(math.isfinite(value) for value in xyz):
                parser.error(f"non-finite coordinates for --asset_xyz {name!r}")
            fixed_asset_xyz[name] = xyz
    asset_layout_file = (
        args.asset_layout_file.resolve() if args.asset_layout_file is not None else None
    )
    if asset_layout_file is not None:
        if not asset_layout_file.is_file():
            raise FileNotFoundError(asset_layout_file)
        try:
            fixed_asset_xyz = load_layout_manifest(
                asset_layout_file,
                required_assets=scene_assets,
                expected_num_envs=args.num_envs,
            )
        except ValueError as exc:
            parser.error(str(exc))
    object_positions: dict[str, object] | None = None
    if args.random_xy:
        object_positions = sample_red_blue_batch(
            args.num_envs,
            seed=args.seed if args.random_xy_seed is None else args.random_xy_seed,
        )
    elif args.blue_xyz is not None:
        object_positions = {
            "blue": [float(value) for value in args.blue_xyz],
            "red": [float(value) for value in args.red_xyz],
        }
    return fixed_asset_xyz, asset_layout_file, object_positions


def _resolve_artifacts(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, Path], Path | None, Path | None, dict[Skill, str] | None]:
    checkpoints = {
        "GRASP": args.grasp_checkpoint.resolve(),
        "LIFT": args.lift_checkpoint.resolve(),
        "TRANSPORT": args.transport_checkpoint.resolve(),
        "ALIGN": args.align_checkpoint.resolve(),
        "DESCEND": args.descend_checkpoint.resolve(),
        "RELEASE_STABILIZE": args.release_checkpoint.resolve(),
        "RETREAT": args.retreat_checkpoint.resolve(),
    }
    reach_checkpoint = (
        args.reach_checkpoint.resolve() if args.reach_checkpoint is not None else None
    )
    if reach_checkpoint is not None and not reach_checkpoint.is_file():
        raise FileNotFoundError(reach_checkpoint)
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    artifact_lock = args.artifact_lock.resolve() if args.artifact_lock is not None else None
    expected_hashes = None
    if artifact_lock is not None:
        if not artifact_lock.is_file():
            raise FileNotFoundError(artifact_lock)
        try:
            expected_hashes = load_action_checkpoint_hashes(artifact_lock)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --artifact_lock: {exc}")
    return checkpoints, reach_checkpoint, artifact_lock, expected_hashes


def _prepare_output_directory(path: Path | None, *, label: str) -> Path | None:
    if path is None:
        return None
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {label}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def parse_evaluation_plan(
    app_launcher_class: Any,
    argv: list[str] | None = None,
) -> EvaluationPlan:
    parser = build_parser(app_launcher_class)
    args = parser.parse_args(argv)
    _load_command(args, parser)
    try:
        scene_assets = scene_cube_assets(args.scene_cube_count)
    except ValueError as exc:
        parser.error(str(exc))
    if args.use_vision and args.scene_cube_count > 4:
        parser.error("the compact RGB-D benchmark supports cube_1 through cube_4")
    skill_translation_limits = _validate_numeric_args(args, parser)
    observer_camera_model = _load_observer(args, parser)
    task_pairs, command_text, language_service, label_to_asset = _resolve_task(
        args, parser, scene_assets
    )
    fixed_asset_xyz, asset_layout_file, object_positions = _resolve_layout(
        args, parser, scene_assets
    )
    checkpoints, reach_checkpoint, artifact_lock, expected_hashes = _resolve_artifacts(
        args, parser
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demonstration_dir = _prepare_output_directory(
        args.demonstration_dir, label="demonstrations"
    )
    if bool(args.dagger_label_skills) != (args.dagger_dir is not None):
        parser.error("--dagger_dir and at least one --dagger_label_skills value are required together")
    dagger_dir = _prepare_output_directory(args.dagger_dir, label="DAgger data")
    video_path = (
        args.video_path.resolve() if args.video_path is not None else output_path.with_suffix(".mp4")
    )
    return EvaluationPlan(
        args=args,
        scene_assets=scene_assets,
        skill_translation_limits=skill_translation_limits,
        task_pairs=task_pairs,
        command_text=command_text,
        language_service=language_service,
        label_to_asset=label_to_asset,
        fixed_asset_xyz=fixed_asset_xyz,
        asset_layout_file=asset_layout_file,
        checkpoints=checkpoints,
        reach_checkpoint=reach_checkpoint,
        artifact_lock=artifact_lock,
        expected_checkpoint_hashes=expected_hashes,
        output_path=output_path,
        demonstration_dir=demonstration_dir,
        dagger_dir=dagger_dir,
        video_path=video_path,
        object_positions=object_positions,
        observer_camera_model=observer_camera_model,
    )


def main() -> None:
    from isaaclab.app import AppLauncher

    plan = parse_evaluation_plan(AppLauncher)
    from .episode_runner import run_evaluation

    run_evaluation(plan, AppLauncher)


__all__ = [
    "EvaluationPlan",
    "build_parser",
    "main",
    "parse_evaluation_plan",
    "scene_cube_assets",
]
