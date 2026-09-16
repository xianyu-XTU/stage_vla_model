"""Evaluate an audited V7 vision-language-action chain in Isaac Lab.

Language and RGB-D observations are validated by V7 services.  Every learned
action is produced by ``StageVLAPipeline.act`` before the migrated simulator
adapter applies its environment-specific safety projection.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

V7_ROOT = Path(__file__).resolve().parents[2]
VENDORED_V5_ROOT = V7_ROOT / "vendor" / "stage_vla_v5"
V5_ROOT = Path(
    os.environ.get(
        "STAGE_VLA_V5_ROOT",
        VENDORED_V5_ROOT if VENDORED_V5_ROOT.is_dir() else V7_ROOT.parent / "stage_vla_v5",
    )
).resolve()
sys.path.insert(0, str(V5_ROOT))
sys.path.insert(0, str(V7_ROOT / "src"))

from stage_vla_v7.contracts import ObjectDetection, SceneState, SKILL_SEQUENCE, Skill
from stage_vla_v7.action.evaluation import legacy_vectorized_skill_success
from stage_vla_v7.language import DeterministicLanguageProvider, LanguageRequest, LanguageService
from stage_vla_v7.orchestration import StageVLAPipeline, default_cube_catalog
from stage_vla_v7.simulation.isaac_lab import (
    OBSERVER_CAMERA_NAME,
    VISION_CAMERA_NAME,
    IsaacCameraAdapter,
    PipelineActionSource,
    build_torchscript_cube_service,
    create_environment,
)
from stage_vla_v7.simulation.config import CameraSpec
from stage_vla_v7.simulation.models import ObserverCameraModel
from stage_vla_v7.simulation.randomization import (
    load_layout_manifest,
    sample_red_blue_batch,
)
from stage_vla_v7.simulation.recording import (
    FrameCapture,
    RecordingConfig,
    VideoRecorder,
)
from stage_vla_v7.vision import (
    LegacyDetectorAdapter,
    StaticVisionProvider,
    VisionRequest,
    VisionService,
)

from tools.train_known_size_grasp import _load_object_metadata, _load_size
from tools.evaluation.result_writer import write_json_result


EXTRA_CUBE_COLORS = (
    (220, 190, 40),
    (170, 80, 210),
    (40, 190, 190),
    (235, 120, 40),
)

ASSET_TO_VISION_LABEL = {
    "cube_1": "blue_cube",
    "cube_2": "red_cube",
    "cube_3": "green_cube",
    "cube_4": "yellow_cube",
}


def build_evaluation_camera_specs(
    *,
    use_vision: bool,
    observer: ObserverCameraModel | None,
    video_width: int | None = None,
    video_height: int | None = None,
    video_fps: float | None = None,
) -> tuple[CameraSpec, ...]:
    """Build independent model-input and observer camera declarations."""
    specs: list[CameraSpec] = []
    if use_vision:
        specs.append(CameraSpec(
            name=VISION_CAMERA_NAME,
            role="vision",
            width=128,
            height=128,
            data_types=("rgb", "distance_to_image_plane"),
        ))
    if observer is not None:
        specs.append(observer.to_camera_spec(
            width=video_width,
            height=video_height,
            fps=video_fps,
        ))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Vision and Observer camera IDs must be distinct")
    return tuple(specs)


def _camera_cfg_transform(quat: np.ndarray, position: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the camera-to-environment-root transform from an Isaac ROS pose."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("camera quaternion must be finite xyzw")
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return transform


def scene_cube_assets(count: int) -> tuple[str, ...]:
    """Return stable Isaac asset names for a configurable cube scene."""
    if not 3 <= int(count) <= 3 + len(EXTRA_CUBE_COLORS):
        raise ValueError(f"scene cube count must be in [3,{3 + len(EXTRA_CUBE_COLORS)}]")
    return tuple(f"cube_{index}" for index in range(1, int(count) + 1))


def load_action_checkpoint_hashes(path: Path) -> dict[Skill, str]:
    """Load and validate all eight Skill records from the artifact lock."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage_vla_v7.external_artifacts.v2":
        raise ValueError("unsupported artifact lock schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifact lock must contain an artifacts object")
    result: dict[Skill, str] = {}
    for skill in SKILL_SEQUENCE:
        record = artifacts.get(skill.value.lower())
        dimension = 52 if skill is Skill.REACH else 55
        if (
            not isinstance(record, dict)
            or record.get("skill") != skill.value
            or record.get("model_type") != "torchscript_parameter_policy"
            or record.get("observation_dim") != dimension
            or record.get("action_dim") != 5
        ):
            raise ValueError(f"invalid artifact lock record for {skill.value}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid checkpoint SHA256 for {skill.value}")
        result[skill] = digest
    return result


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--reach_checkpoint", type=Path, default=None,
                   help="optional independent REACH policy; omit for geometric reference")
    p.add_argument(
        "--reach_model_type", default="independent_bc_policy",
        help="controller type recorded when --reach_checkpoint is supplied",
    )
    p.add_argument("--grasp_checkpoint", type=Path, required=True)
    p.add_argument("--lift_checkpoint", type=Path, required=True)
    p.add_argument("--transport_checkpoint", type=Path, required=True)
    p.add_argument("--align_checkpoint", type=Path, required=True)
    p.add_argument("--descend_checkpoint", type=Path, required=True)
    p.add_argument("--release_checkpoint", type=Path, required=True)
    p.add_argument("--retreat_checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--artifact_lock",
        type=Path,
        default=None,
        help="optional V7 artifact lock that must cover and match all eight policies",
    )
    p.add_argument(
        "--use_vision", action="store_true",
        help="replace object/support positions in policy observations with RGB-D detections",
    )
    p.add_argument(
        "--require_v7_chain",
        action="store_true",
        help="fail the result unless vision, language, and every learned action use V7 services",
    )
    p.add_argument("--camera_calibration", type=Path, default=None)
    p.add_argument("--vision_confidence_floor", type=float, default=0.5)
    p.add_argument("--vision_min_pixels", type=int, default=16)
    p.add_argument("--geometry_bias_m", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument(
        "--demonstration_dir", type=Path, default=None,
        help="save successful reference observations/actions for per-skill BC",
    )
    p.add_argument(
        "--dagger_dir", type=Path, default=None,
        help="save teacher labels on states visited by model-controlled skills",
    )
    p.add_argument(
        "--dagger_label_skills", nargs="*", default=(),
        choices=("REACH", "GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND",
                 "RELEASE_STABILIZE", "RETREAT"),
        help="model still acts; geometric teacher only labels visited states",
    )
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument(
        "--scene_cube_count", type=int, default=3,
        help="number of cube assets instantiated in the scene (3-7)",
    )
    p.add_argument("--reach_steps", type=int, default=600)
    p.add_argument("--reach_stable_steps", type=int, default=3,
                   help="consecutive geometry-valid steps required to finish REACH")
    p.add_argument("--reach_refine_steps", type=int, default=12,
                   help="open-gripper geometry corrections after all environments reach")
    p.add_argument(
        "--reach_recovery_steps", type=int, default=0,
        help="optional geometric REACH recovery after a learned REACH exhausts its horizon",
    )
    p.add_argument("--grasp_steps", type=int, default=80)
    p.add_argument("--lift_steps", type=int, default=100)
    p.add_argument("--transport_steps", type=int, default=180)
    p.add_argument("--align_steps", type=int, default=180)
    p.add_argument("--descend_steps", type=int, default=100)
    p.add_argument("--release_steps", type=int, default=180)
    p.add_argument("--retreat_steps", type=int, default=220)
    p.add_argument(
        "--reference_skills", nargs="*", default=(),
        choices=("GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND",
                 "RELEASE_STABILIZE", "RETREAT"),
        help="diagnostic only: replace selected checkpoints with geometry reference actions",
    )
    p.add_argument("--grasp_stable_steps", type=int, default=3)
    p.add_argument("--lift_stable_steps", type=int, default=3)
    p.add_argument("--transport_stable_steps", type=int, default=3)
    p.add_argument("--align_stable_steps", type=int, default=3)
    p.add_argument("--descend_stable_steps", type=int, default=3)
    p.add_argument("--release_stable_steps", type=int, default=5)
    p.add_argument("--retreat_stable_steps", type=int, default=20)
    p.add_argument("--grasp_to_lift_settle_steps", type=int, default=10)
    p.add_argument("--lift_translation_limit_m", type=float, default=None,
                   help="maximum Cartesian LIFT translation per control step")
    p.add_argument("--pregrasp_contact_descent_m", type=float, default=0.0,
                   help="deterministic REACH->GRASP contact-depth conditioner")
    p.add_argument("--lift_to_transport_settle_steps", type=int, default=10)
    p.add_argument("--transport_to_align_settle_steps", type=int, default=10)
    p.add_argument("--align_to_descend_settle_steps", type=int, default=10)
    p.add_argument(
        "--final_stack_settle_steps", type=int, default=0,
        help="open-gripper physics-only settling before the final multi-object stack check",
    )
    p.add_argument("--transport_xy_m", type=float, default=0.045)
    p.add_argument("--transport_translation_limit_m", type=float, default=None)
    p.add_argument("--align_translation_limit_m", type=float, default=None)
    p.add_argument("--descend_translation_limit_m", type=float, default=None)
    p.add_argument("--release_translation_limit_m", type=float, default=None)
    p.add_argument("--retreat_translation_limit_m", type=float, default=None)
    p.add_argument("--align_xy_m", type=float, default=0.010)
    p.add_argument("--align_height_m", type=float, default=0.0618)
    p.add_argument("--align_height_tolerance_m", type=float, default=0.015)
    p.add_argument("--descend_height_m", type=float, default=0.0468)
    p.add_argument("--descend_height_tolerance_m", type=float, default=0.002)
    p.add_argument("--stack_xy_m", type=float, default=0.040)
    p.add_argument("--stack_height_tolerance_m", type=float, default=0.010)
    p.add_argument("--speed_mps", type=float, default=0.05)
    p.add_argument("--physical_height_tolerance_m", type=float, default=0.015)
    p.add_argument("--retreat_distance_m", type=float, default=0.100)
    p.add_argument("--retreat_height_m", type=float, default=0.080)
    p.add_argument("--effort_limit", type=float, default=40.0)
    p.add_argument(
        "--carry_leveling_max_angle_rad", type=float, default=0.0,
        help="bounded roll/pitch correction that keeps the parallel-jaw axis level",
    )
    p.add_argument(
        "--pregrasp_yaw_tolerance_rad", type=float, default=0.005,
        help="maximum box-edge yaw error before GRASP is allowed to close",
    )
    p.add_argument("--lift_arm_warmup_steps", type=int, default=0)
    p.add_argument("--lift_arm_warmup_scale", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2012)
    p.add_argument("--object_asset", default="cube_2",
                   help="Isaac scene asset assigned to the manipulated-object role")
    p.add_argument("--support_asset", default="cube_1",
                   help="Isaac scene asset assigned to the support role")
    p.add_argument(
        "--task_pairs", nargs="+", default=None, metavar="OBJECT:SUPPORT",
        help=("ordered object/support relations executed in one simulator episode; "
              "for example cube_1:cube_3 cube_2:cube_1"),
    )
    p.add_argument(
        "--command",
        default=None,
        help=("natural-language or DSL stack command; when supplied it owns task "
              "planning and must resolve to assets in the configured scene"),
    )
    p.add_argument(
        "--blue_xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="fixed blue cube position in the local environment frame",
    )
    p.add_argument(
        "--red_xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="fixed red cube position in the local environment frame",
    )
    p.add_argument(
        "--asset_xyz", nargs=4, action="append", default=None,
        metavar=("ASSET", "X", "Y", "Z"),
        help="fixed local position for one scene asset; repeat for multi-object layouts",
    )
    p.add_argument(
        "--asset_layout_file", type=Path, default=None,
        help="JSON layout batch with one XYZ row per asset and parallel environment",
    )
    p.add_argument(
        "--random_xy", action="store_true",
        help="sample one blue/red XY pair per environment; Z remains 0.0203 m",
    )
    p.add_argument(
        "--random_xy_seed", type=int, default=None,
        help="seed for reproducible XY pair generation (defaults to --seed)",
    )
    p.add_argument("--video", action="store_true",
                   help="record one environment's RGB trajectory to an MP4")
    p.add_argument("--video_path", type=Path, default=None,
                   help="output MP4 path; defaults to output JSON path with .mp4")
    p.add_argument("--video_env", type=int, default=0,
                   help="environment index to record when --video is enabled")
    p.add_argument(
        "--video_env_only", action="store_true",
        help="gate a recording run on the recorded environment instead of every parallel environment",
    )
    p.add_argument("--video_width", type=int, default=None)
    p.add_argument("--video_height", type=int, default=None)
    p.add_argument("--video_fps", type=float, default=None)
    p.add_argument(
        "--observer_camera_config",
        type=Path,
        default=V7_ROOT / "config" / "simulation" / "observer_camera.json",
        help="third-person camera configuration; never used as Vision input",
    )
    recording_mode = p.add_mutually_exclusive_group()
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
    p.set_defaults(recording_strict=True)
    p.add_argument(
        "--trace_env", type=int, default=None,
        help="optional environment index for per-step physical diagnostics",
    )
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.command is None:
        encoded_command = os.environ.get("STAGE_VLA_COMMAND_UTF8_BASE64")
        if encoded_command:
            try:
                args.command = base64.b64decode(
                    encoded_command, validate=True
                ).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
                p.error(f"invalid STAGE_VLA_COMMAND_UTF8_BASE64: {exc}")
        else:
            environment_command = os.environ.get("STAGE_VLA_COMMAND")
            if environment_command:
                args.command = environment_command
    try:
        scene_assets = scene_cube_assets(args.scene_cube_count)
    except ValueError as exc:
        p.error(str(exc))
    if args.use_vision and args.scene_cube_count > 4:
        p.error("the compact RGB-D benchmark supports cube_1 through cube_4")
    if not 0.0 < args.vision_confidence_floor <= 1.0:
        p.error("--vision_confidence_floor must be in (0,1]")
    if args.vision_min_pixels < 1:
        p.error("--vision_min_pixels must be positive")
    from stage_vla.stages.object_motion import motion_profile_from_config

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
        args.num_envs, args.reach_steps, args.reach_stable_steps,
        args.grasp_steps, args.lift_steps,
        args.transport_steps, args.align_steps, args.descend_steps,
        args.release_steps, args.retreat_steps, args.grasp_stable_steps,
        args.lift_stable_steps, args.transport_stable_steps, args.align_stable_steps,
        args.descend_stable_steps, args.release_stable_steps, args.retreat_stable_steps,
    )
    if min(positive) < 1:
        p.error("counts, horizons and stable steps must be positive")
    settles = (
        args.grasp_to_lift_settle_steps, args.lift_to_transport_settle_steps,
        args.transport_to_align_settle_steps, args.align_to_descend_settle_steps,
        args.final_stack_settle_steps,
    )
    if min(*settles, args.reach_recovery_steps) < 0:
        p.error("settle and recovery steps must be non-negative")
    if not 0.0 <= args.pregrasp_contact_descent_m <= 0.005:
        p.error("pregrasp_contact_descent_m must be in [0, 0.005]")
    motion_limits = tuple(
        getattr(args, field) for field in motion_profile.__dataclass_fields__
    )
    if min(args.physical_height_tolerance_m, *motion_limits) <= 0:
        p.error("physical tolerances and translation limits must be positive")
    if args.video and not 0 <= args.video_env < args.num_envs:
        p.error("video_env must be a valid environment index")
    if args.video_env_only and not args.video:
        p.error("--video_env_only requires --video")
    observer_camera_model = None
    if args.video:
        observer_config = args.observer_camera_config.resolve()
        if not observer_config.is_file():
            p.error(f"observer camera config does not exist: {observer_config}")
        try:
            observer_camera_model = ObserverCameraModel.from_json(observer_config)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            p.error(f"invalid observer camera config: {exc}")
        if args.video_width is None:
            args.video_width = observer_camera_model.width
        if args.video_height is None:
            args.video_height = observer_camera_model.height
        if args.video_fps is None:
            args.video_fps = observer_camera_model.fps
        if min(args.video_width, args.video_height) < 1:
            p.error("video dimensions must be positive")
        if args.video_fps <= 0:
            p.error("video_fps must be positive")
    if args.trace_env is not None and not 0 <= args.trace_env < args.num_envs:
        p.error("trace_env must be in [0, num_envs)")
    if (args.blue_xyz is None) != (args.red_xyz is None):
        p.error("--blue_xyz and --red_xyz must be supplied together")
    if args.random_xy and args.blue_xyz is not None:
        p.error("--random_xy cannot be combined with --blue_xyz/--red_xyz")
    if args.asset_xyz is not None and (args.random_xy or args.blue_xyz is not None):
        p.error("--asset_xyz cannot be combined with pair position overrides")
    if args.asset_layout_file is not None and (
        args.asset_xyz is not None or args.random_xy or args.blue_xyz is not None
    ):
        p.error("--asset_layout_file cannot be combined with other position overrides")
    if args.object_asset == args.support_asset:
        p.error("--object_asset and --support_asset must differ")
    if args.object_asset not in scene_assets or args.support_asset not in scene_assets:
        p.error("--object_asset and --support_asset must exist in the configured cube scene")
    task_pairs = [(args.object_asset, args.support_asset)]
    if args.task_pairs is not None:
        task_pairs = []
        valid_assets = set(scene_assets)
        for value in args.task_pairs:
            parts = value.split(":")
            if len(parts) != 2 or parts[0] not in valid_assets or parts[1] not in valid_assets:
                p.error(f"invalid --task_pairs value {value!r}; expected cube_N:cube_N")
            if parts[0] == parts[1]:
                p.error(f"task pair roles must differ: {value!r}")
            task_pairs.append((parts[0], parts[1]))
    language_service = LanguageService(DeterministicLanguageProvider())
    label_to_asset = {
        label: asset for asset, label in ASSET_TO_VISION_LABEL.items() if asset in scene_assets
    }
    available_labels = tuple(label_to_asset)
    if args.command is not None:
        try:
            language_result = language_service.interpret(
                LanguageRequest(args.command, available_labels)
            )
            command_pairs = [
                (label_to_asset[relation.object_label], label_to_asset[relation.support_label])
                for relation in language_result.plan.execution_relations
            ]
        except (KeyError, ValueError, LookupError) as exc:
            p.error(f"invalid --command: {exc}")
        if args.task_pairs is not None and command_pairs != task_pairs:
            p.error("--command and --task_pairs resolve to different execution plans")
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
        p.error("fixed/random pair overrides currently require cube_2 -> cube_1 roles")
    if len(task_pairs) > 1 and (args.random_xy or args.blue_xyz is not None):
        p.error("multi-relation evaluation cannot use fixed/random pair overrides")
    fixed_asset_xyz = None
    if args.asset_xyz is not None:
        fixed_asset_xyz = {}
        for name, *coordinates in args.asset_xyz:
            if name not in set(scene_assets):
                p.error(f"invalid asset name in --asset_xyz: {name!r}")
            if name in fixed_asset_xyz:
                p.error(f"duplicate --asset_xyz entry: {name!r}")
            try:
                xyz = tuple(float(value) for value in coordinates)
            except ValueError:
                p.error(f"invalid coordinates for --asset_xyz {name!r}")
            if not all(math.isfinite(value) for value in xyz):
                p.error(f"non-finite coordinates for --asset_xyz {name!r}")
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
            p.error(str(exc))

    checkpoints = {
        "GRASP": args.grasp_checkpoint.resolve(),
        "LIFT": args.lift_checkpoint.resolve(),
        "TRANSPORT": args.transport_checkpoint.resolve(),
        "ALIGN": args.align_checkpoint.resolve(),
        "DESCEND": args.descend_checkpoint.resolve(),
        "RELEASE_STABILIZE": args.release_checkpoint.resolve(),
        "RETREAT": args.retreat_checkpoint.resolve(),
    }
    reach_checkpoint = args.reach_checkpoint.resolve() if args.reach_checkpoint is not None else None
    if reach_checkpoint is not None and not reach_checkpoint.is_file():
        raise FileNotFoundError(reach_checkpoint)
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    artifact_lock = args.artifact_lock.resolve() if args.artifact_lock is not None else None
    expected_checkpoint_hashes = None
    if artifact_lock is not None:
        if not artifact_lock.is_file():
            raise FileNotFoundError(artifact_lock)
        try:
            expected_checkpoint_hashes = load_action_checkpoint_hashes(artifact_lock)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            p.error(f"invalid --artifact_lock: {exc}")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    demonstration_dir = (
        args.demonstration_dir.resolve() if args.demonstration_dir is not None else None
    )
    if demonstration_dir is not None:
        if demonstration_dir.exists() and any(demonstration_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty demonstrations: {demonstration_dir}"
            )
        demonstration_dir.mkdir(parents=True, exist_ok=True)
    dagger_dir = args.dagger_dir.resolve() if args.dagger_dir is not None else None
    if bool(args.dagger_label_skills) != (dagger_dir is not None):
        p.error("--dagger_dir and at least one --dagger_label_skills value are required together")
    if dagger_dir is not None:
        if dagger_dir.exists() and any(dagger_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty DAgger data: {dagger_dir}")
        dagger_dir.mkdir(parents=True, exist_ok=True)
    video_path = (args.video_path.resolve() if args.video_path is not None
                  else out.with_suffix(".mp4"))
    if args.video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    object_positions = None
    if args.random_xy:
        object_positions = sample_red_blue_batch(
            args.num_envs,
            seed=args.seed if args.random_xy_seed is None else args.random_xy_seed,
        )
    elif args.blue_xyz is not None:
        object_positions = {
            "blue": [float(v) for v in args.blue_xyz],
            "red": [float(v) for v in args.red_xyz],
        }

    app = AppLauncher(args).app
    raw = env = None
    video_recorder = None
    recording_result = None
    try:
        import torch
        from stage_vla.action_output import ActionOutputModule
        from stage_vla.data.v4_2_runtime import read_depth_m_batch, read_rgb_u8_batch
        from stage_vla.envs.state_readers import to_torch
        from stage_vla.rl.reach_policy import measure_reach_state, reach_observation, reach_raw_action
        from stage_vla.rl.v5_skill_contracts import reference_action
        from stage_vla.rl.known_size_grasp_vecenv import KnownSizeGraspVecEnv
        from stage_vla.rl.known_size_grasp import pressure_tracking_ok
        from stage_vla.rl.skill_action_safety import (
            jaw_leveling_axis_angle,
            object_upright_tilt_rad,
            project_pregrasp_edge_alignment,
        )
        from stage_vla.rl.skill_demonstrations import SkillDemonstrationBuffer
        from stage_vla.stages.grasp_geometry import (
            grasp_target_position,
            local_width_ratio,
            parallel_jaw_yaw_error,
            profile_from_config,
        )
        from stage_vla.vision import CameraCalibration, CompactColorDepthDetector
        skill_success = legacy_vectorized_skill_success

        size = _load_size(args.config.resolve())
        object_geometry, object_color_rgb = _load_object_metadata(args.config.resolve())
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
        env = KnownSizeGraspVecEnv(
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
        env.auto_reset = False
        env.physical_cfg = replace(
            env.physical_cfg, height_tolerance_m=float(args.physical_height_tolerance_m)
        )

        vision_stats = {
            "enabled": bool(args.use_vision),
            "frames": 0,
            "invalid_frames": 0,
            "v7_service_calls": 0,
            "missing_by_asset": {name: 0 for name in scene_assets},
        }
        vision_detector = None
        vision_camera = None
        vision_calibration = None
        vision_service = None
        vision_origins = None
        vision_tracked: dict[str, np.ndarray] = {}
        vision_seed_valid = np.ones(args.num_envs, dtype=bool)
        if args.use_vision:
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
            runtime_transform = _camera_cfg_transform(
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
                LegacyDetectorAdapter(
                    vision_detector,
                    name="v7-compact-color-depth",
                    version="1",
                    method="detect_scene",
                    method_kwargs={
                        "calibration": vision_calibration,
                        "labels": tuple(
                            ASSET_TO_VISION_LABEL[name] for name in scene_assets
                        ),
                    },
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
            rgb = read_rgb_u8_batch(raw.unwrapped, camera_name=VISION_CAMERA_NAME)
            depth = read_depth_m_batch(raw.unwrapped, camera_name=VISION_CAMERA_NAME)
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
                    if detection is None:
                        vision_stats["missing_by_asset"][asset_name] += 1
                    else:
                        predicted[asset_name][env_index] = np.asarray(
                            detection.position_xyz_m, dtype=np.float32
                        )
            valid = np.ones(args.num_envs, dtype=bool)
            result = {}
            for asset_name, local_xyz in predicted.items():
                observed = np.isfinite(local_xyz).all(axis=1)
                previous = vision_tracked.get(asset_name)
                if previous is None:
                    vision_tracked[asset_name] = local_xyz.copy()
                else:
                    local_xyz = np.where(observed[:, None], local_xyz, previous)
                    vision_tracked[asset_name] = np.where(
                        observed[:, None], local_xyz, previous
                    )
                asset_valid = np.isfinite(local_xyz).all(axis=1)
                valid &= asset_valid
                if not bool(asset_valid.all()):
                    oracle_world = to_torch(
                        raw.unwrapped.scene[asset_name].data.root_pos_w
                    )[..., :3]
                    oracle_local = (
                        oracle_world - vision_origins
                    ).detach().cpu().numpy()
                    local_xyz = np.where(
                        asset_valid[:, None], local_xyz, oracle_local
                    )
                result[asset_name] = (
                    torch.as_tensor(local_xyz, device=env.device, dtype=torch.float32)
                    + vision_origins
                )
            vision_stats["frames"] += int(args.num_envs)
            vision_stats["invalid_frames"] += int((~valid).sum())
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
            initial_rgb = read_rgb_u8_batch(
                raw.unwrapped, camera_name=VISION_CAMERA_NAME
            )[0]
            initial_depth = read_depth_m_batch(
                raw.unwrapped, camera_name=VISION_CAMERA_NAME
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
        trace: list[dict[str, object]] = []

        def append_trace(
            *,
            relation_index: int,
            skill: str,
            step: int,
            checkpoint_action=None,
            submitted_action=None,
        ) -> None:
            if args.trace_env is None:
                return
            i = args.trace_env
            measured = env.measured
            rel = measured["red"][i] - measured["blue"][i]
            jaw = measured["right_tip"][i] - measured["left_tip"][i]
            leveling = jaw_leveling_axis_angle(
                measured["left_tip"][i:i + 1],
                measured["right_tip"][i:i + 1],
            )[0]
            row: dict[str, object] = {
                "relation": relation_index + 1,
                "skill": skill,
                "step": int(step),
                "projected_action": env.prev_unit[i].detach().cpu().tolist(),
                "stack_relative_xyz_m": rel.detach().cpu().tolist(),
                "object_quaternion_xyzw": measured["red_quat"][i].detach().cpu().tolist(),
                "upright_tilt_rad": float(object_upright_tilt_rad(
                    measured["red_quat"][i:i + 1]
                )[0]),
                "angular_velocity_radps": measured["angular"][i].detach().cpu().tolist(),
                "angular_speed_radps": float(
                    measured.get("stability_angular_speed", measured["angular"].norm(dim=-1))[i]
                ),
                "instantaneous_angular_speed_radps": float(measured["angular"][i].norm()),
                "control_angular_speed_radps": float(
                    measured.get("control_angular_speed", measured["angular"].norm(dim=-1))[i]
                ),
                "linear_velocity_mps": measured["vel"][i].detach().cpu().tolist(),
                "stability_speed_mps": float(measured["stability_speed"][i]),
                "instantaneous_speed_mps": float(measured["speed"][i]),
                "left_fingertip_position_m": measured["left_tip"][i].detach().cpu().tolist(),
                "right_fingertip_position_m": measured["right_tip"][i].detach().cpu().tolist(),
                "jaw_axis_w": jaw.detach().cpu().tolist(),
                "jaw_height_delta_m": float(jaw[2]),
                "jaw_leveling_axis_angle_rad": leveling.detach().cpu().tolist(),
                "force_n": measured["force"][i].detach().cpu().tolist(),
                "physical_grasp": bool(measured["physical"][i]),
                "stable_steps": int(env.stable_count[i]),
            }
            if checkpoint_action is not None:
                row["checkpoint_action"] = checkpoint_action[i].detach().cpu().tolist()
            if submitted_action is not None:
                row["submitted_action"] = submitted_action[i].detach().cpu().tolist()
            trace.append(row)

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
            )
            if args.use_vision:
                active_mask &= torch.as_tensor(
                    vision_seed_valid,
                    dtype=torch.bool,
                    device=reach_state["ee"].device,
                )
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
                    env.physical_batch.object_size_m,
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
                        env.physical_batch.object_size_m,
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
                        finished=reached,
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
                    raw.step(reach_raw_action(action))
                    reach_reset |= bool(raw.unwrapped.reset_buf.any())
                    if reach_reset:
                        raise RuntimeError("simulator reset during REACH")
                    capture_video_frame(f"{label} REACH", step)
                    reach_state = measure_policy_reach_state(
                        object_asset, support_asset
                    )
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
                        finished=reached,
                        translation_limit_m=0.005,
                        yaw_limit_rad=0.02,
                    )
                    action = output.command
                    action, _aligning, _yaw_error = project_reach_safety(
                        action, output.active
                    )
                    raw.step(reach_raw_action(action))
                    reach_reset |= bool(raw.unwrapped.reset_buf.any())
                    if reach_reset:
                        raise RuntimeError("simulator reset during REACH recovery")
                    capture_video_frame(f"{label} REACH recovery", recovery_step)
                    reach_state = measure_policy_reach_state(
                        object_asset, support_asset
                    )
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
                return env._obs(), {
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

            # A learned REACH has already satisfied the consecutive-step gate;
            # hand it off immediately.  Reference collection keeps its fixed
            # refinement to match the historical demonstration distribution.
            refine_steps = args.reach_refine_steps if not learned_reach else 0
            with torch.inference_mode():
                for refine_step in range(1, refine_steps + 1):
                    reach_obs = reach_observation(reach_state, reach_previous)
                    output = output_module.emit(
                        "REACH",
                        reach_obs,
                        reference_state=reach_state,
                        translation_limit_m=0.005,
                        yaw_limit_rad=0.02,
                    )
                    action = output.command
                    if demonstration_dir is not None and not learned_reach:
                        demonstration_rows.append("REACH", reach_obs, action)
                    action, _aligning, _yaw_error = project_reach_safety(
                        action, torch.ones_like(reached)
                    )
                    raw.step(reach_raw_action(action))
                    if raw.unwrapped.reset_buf.any():
                        raise RuntimeError("simulator reset during REACH refinement")
                    capture_video_frame(f"{label} REACH refine", refine_step)
                    reach_state = measure_policy_reach_state(
                        object_asset, support_asset
                    )
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
                        env.physical_batch.object_size_m,
                        reach_state["left_tip"],
                        reach_state["right_tip"],
                    ).detach().cpu().tolist()
                )
            conditioners = []
            if args.pregrasp_contact_descent_m:
                contact_action = torch.zeros(
                    args.num_envs, 7, device=reach_state["ee"].device
                )
                contact_action[:, 2] = -float(args.pregrasp_contact_descent_m) / 0.005
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
            env.measured = env._measure()
            env.entry_red_z[:] = env.measured["red"][:, 2]
            env.prev_force[:] = env.measured["force"]
            inject_visual_roles(object_asset, support_asset)
            return env._obs(), {
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
            )
            env.finished[~active_mask] = True
            for step in range(1, horizon + 1):
                inject_visual_roles(object_asset, support_asset)
                obs = env._obs()
                reference_state = dict(env.measured)
                reference_state["held"] = env.measured["physical"]
                reference_state["stack_height"] = torch.full(
                    (args.num_envs,), args.descend_height_m, device=env.device
                )
                output = output_module.emit(
                    skill,
                    obs["policy"],
                    reference_state=reference_state,
                    finished=env.finished,
                    translation_limit_m=skill_translation_limits.get(skill, 0.005),
                    yaw_limit_rad=0.02,
                    include_reference=skill in args.dagger_label_skills,
                )
                action = output.command
                if skill != "REACH":
                    from stage_vla.rl.skill_action_safety import (
                        hold_finished_skill_action,
                    )
                    action = hold_finished_skill_action(
                        skill, action, env.finished, env.prev_unit
                    )
                active = output.active
                if output.source == "reference":
                    if demonstration_dir is not None and bool(active.any()):
                        demonstration_rows.append(
                            skill, obs["policy"][active], action[active]
                        )
                else:
                    if skill in args.dagger_label_skills and bool(active.any()):
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
                append_trace(
                    relation_index=relation_index,
                    skill=skill,
                    step=step,
                    checkpoint_action=output.command,
                    submitted_action=action,
                )
                for i in (record_done & active_mask).nonzero(as_tuple=False).flatten().tolist():
                    rel = env.measured["red"][i] - env.measured["blue"][i]
                    rows[i] = {
                        "env": i, "steps": step,
                        "success": bool(env.last_success[i]),
                        "failure": bool(env.last_failure[i]),
                        "timeout": bool(env.last_timeout[i]),
                        "physical_grasp": bool(env.measured["physical"][i]),
                        "pressure_ok": bool(pressure_tracking_ok(
                            env.measured["force"][i].unsqueeze(0),
                            env.gripper_action.target_force_n[i].unsqueeze(0), cfg=size
                        )[0]),
                        "force_n": env.measured["force"][i].detach().cpu().tolist(),
                        "target_force_n": float(env.gripper_action.target_force_n[i]),
                        "gripper_joint_m": env.measured["grip"][i].detach().cpu().tolist(),
                        "gripper_target_m": env.gripper_action.processed_actions[i].detach().cpu().tolist(),
                        "fingertip_gap_m": float(env.measured["fingertip_gap_m"][i]),
                        "radial_error_m": float(env.measured["radial_error_m"][i]),
                        "left_height_error_m": float(env.measured["left_height_error_m"][i]),
                        "right_height_error_m": float(env.measured["right_height_error_m"][i]),
                        "stack_xy_m": float(rel[:2].norm()),
                        "stack_relative_height_m": float(rel[2]),
                        "lift_target_height_m": float(env.lift_target_height_m[i]),
                        "stability_speed_mps": float(env.measured["stability_speed"][i]),
                        "instantaneous_speed_mps": float(env.measured["speed"][i]),
                        "angular_speed_radps": float(
                            env.measured["stability_angular_speed"][i]
                        ),
                        "instantaneous_angular_speed_radps": float(
                            env.measured["instantaneous_angular_speed"][i]
                        ),
                        "control_angular_speed_radps": float(
                            env.measured["control_angular_speed"][i]
                        ),
                        "stable_steps": int(env.stable_count[i]),
                        "first_action": first_action[i].tolist(),
                        "terminal_action": action[i].detach().cpu().tolist(),
                    }
                if bool(env.finished[active_mask].all()):
                    break
            return obs, rows

        def handoff(obs, skill: str, horizon: int, stable_steps: int, active_mask):
            before = env._measure()
            env.skill = skill
            env.mark_continuous_handoff()
            env.arm_locked = False
            env.episode_steps = int(horizon)
            env.max_episode_length = int(horizon)
            env.stable_steps = int(stable_steps)
            env.steps.zero_(); env.stable_count.zero_(); env.finished.zero_()
            env.last_success.zero_(); env.last_failure.zero_(); env.last_timeout.zero_()
            env.finished[~active_mask] = True
            after = env._measure()
            exact = all(torch.equal(before[key], after[key]) for key in before)
            if not exact:
                raise RuntimeError(f"physical state changed during handoff to {skill}")
            env.measured = after
            if skill == "LIFT":
                env.update_lift_target_from_support()
            return env._obs(), exact

        def settle(obs, steps: int, label: str, relation_index: int):
            if not steps:
                return obs
            action = torch.zeros(args.num_envs, 5, device=env.device)
            action[:, 4] = -1.0
            for _ in range(steps):
                obs, _reward, _done, _extras = env.step(action)
                if bool(env.unwrapped.reset_buf.any()):
                    raise RuntimeError(f"simulator reset during {label} settle")
                capture_video_frame(label, _ + 1)
                append_trace(
                    relation_index=relation_index,
                    skill=f"{env.skill}_SETTLE",
                    step=_ + 1,
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
        reach_reference_recovery_used = any(
            relation.get("reach_recovery_steps_attempted", 0) > 0
            for relation in relation_results
        )
        v7_audit = pipeline_action_source.audit()
        v7_chain_verified = bool(
            args.use_vision
            and learned_reach
            and not args.reference_skills
            and not reach_reference_recovery_used
            and vision_stats["v7_service_calls"] > 0
            and v7_audit["all_prepared_skills_exercised"]
        )
        if args.require_v7_chain and not v7_chain_verified:
            passed = False
        demonstration_manifest = None
        if demonstration_dir is not None and passed:
            demonstration_manifest = {}
            for skill in demonstration_rows.sample_counts():
                payload = demonstration_rows.payload(
                    skill, object_geometry=object_geometry,
                    grasp_profile={
                        "geometry": grasp_profile.geometry,
                        "height_ratio": grasp_profile.height_ratio,
                        "width_ratio": grasp_profile.width_ratio,
                    },
                )
                path = demonstration_dir / f"{skill.lower()}.pt"
                torch.save(payload, path)
                demonstration_manifest[skill] = {
                    "path": str(path),
                    "samples": int(len(payload["observations"])),
                    "observation_dim": int(payload["observation_dim"]),
                }
            (demonstration_dir / "manifest.json").write_text(
                json.dumps({
                    "status": "complete",
                    "source_result": str(out),
                    "object_geometry": object_geometry,
                    "skills": demonstration_manifest,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        dagger_manifest = None
        if dagger_dir is not None:
            dagger_manifest = {}
            for skill, count in dagger_rows.sample_counts().items():
                if not count:
                    continue
                payload = dagger_rows.payload(
                    skill, object_geometry=object_geometry,
                    collection_mode="model_executed_teacher_labeled",
                    source_checkpoint=str(
                        reach_checkpoint if skill == "REACH" else checkpoints[skill]
                    ),
                    source_result=str(out),
                )
                path = dagger_dir / f"{skill.lower()}.pt"
                torch.save(payload, path)
                dagger_manifest[skill] = {
                    "path": str(path), "samples": count,
                    "observation_dim": int(payload["observation_dim"]),
                }
            (dagger_dir / "manifest.json").write_text(json.dumps({
                "status": "complete" if dagger_manifest else "empty",
                "collection_mode": "model_executed_teacher_labeled",
                "teacher_controlled_environment": reach_reference_recovery_used,
                "teacher_controlled_labeled_transitions": False,
                "teacher_recovery_used_after_policy_horizon": (
                    reach_reference_recovery_used
                ),
                "source_result": str(out),
                "skills": dagger_manifest,
            }, indent=2) + "\n", encoding="utf-8")
        if args.video:
            assert video_recorder is not None
            recording_result = video_recorder.stop()
        primary = relation_results[0]
        single_relation = len(task_pairs) == 1
        top_successes = (
            primary["successes"] if single_relation else {
                f"{item['asset_roles']['object']}->{item['asset_roles']['support']}":
                    item["successes"]
                for item in relation_results
            }
        )
        result = {
            "status": "passed" if passed else "failed",
            "v7_chain": {
                "verified": v7_chain_verified,
                "required": bool(args.require_v7_chain),
                "command": command_text,
                "planned_relations": [
                    {
                        "object": relation.object_label,
                        "support": relation.support_label,
                    }
                    for relation in prepared.language.plan.execution_relations
                ],
                **v7_audit,
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
                **vision_stats,
                "seed_valid": vision_seed_valid.tolist(),
                "backend": "compact_color_depth" if args.use_vision else None,
                "camera_calibration": (
                    str(args.camera_calibration.resolve())
                    if args.use_vision and args.camera_calibration is not None else None
                ),
                "geometry_bias_m": list(args.geometry_bias_m) if args.use_vision else None,
            },
            "episodes": args.num_envs,
            "scene_assets": list(scene_assets),
            "validation_envs": list(validation_envs),
            "object_geometry": object_geometry,
            "asset_roles": primary["asset_roles"],
            "task_pairs": [
                {"object": object_asset, "support": support_asset}
                for object_asset, support_asset in task_pairs
            ],
            "grasp_profile": {
                "geometry": grasp_profile.geometry,
                "height_ratio": grasp_profile.height_ratio,
                "width_ratio": grasp_profile.width_ratio,
            },
            "successes": top_successes,
            "chain_successes": (
                int(overall_alive[list(validation_envs)].sum())
            ),
            "seed_success": overall_alive.detach().cpu().tolist(),
            "mid_episode_resets": int(reset_seen),
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
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
            "artifact_lock": str(artifact_lock) if artifact_lock is not None else None,
            "reach_checkpoint": str(reach_checkpoint) if reach_checkpoint is not None else None,
            "reach_controller": args.reach_model_type if reach_checkpoint is not None else "geometric_reference",
            "reach_refinement_controller": (
                args.reach_model_type if reach_checkpoint is not None
                else "geometric_reference"
            ),
            "reach_recovery_controller": (
                "geometric_reference" if reach_reference_recovery_used else None
            ),
            "pure_policy_actions": bool(
                reach_checkpoint is not None
                and not args.reference_skills
                and not reach_reference_recovery_used
            ),
            "reference_skills": list(args.reference_skills),
            "fixed_object_positions_local_xyz": object_positions,
            "fixed_asset_positions_local_xyz": fixed_asset_xyz,
            "asset_layout_file": (
                str(asset_layout_file) if asset_layout_file is not None else None
            ),
            "random_xy": bool(args.random_xy),
            "random_xy_seed": (args.seed if args.random_xy_seed is None else args.random_xy_seed)
            if args.random_xy else None,
            "handoffs": primary["handoffs"] if single_relation else [
                item["handoffs"] for item in relation_results
            ],
            "interstage_conditioners": (
                primary["interstage_conditioners"] if single_relation else [
                    item["interstage_conditioners"] for item in relation_results
                ]
            ),
            "horizons": {name: horizon for name, horizon, _stable in stage_specs},
            "translation_limits_m": skill_translation_limits,
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
                "angular_speed_source": env.stability_speed_source,
                "retreat_distance_m": args.retreat_distance_m,
                "retreat_height_m": args.retreat_height_m,
            },
            "demonstrations": demonstration_manifest,
            "dagger_labels": dagger_manifest,
            "rows": primary["rows"] if single_relation else None,
            "relations": relation_results,
            "final_stack": final_stack,
            "trace_env": args.trace_env,
            "trace": trace,
        }
        if args.video:
            assert recording_result is not None
            result["video"] = {
                **recording_result.as_dict(),
                "env": int(args.video_env),
                "resolution": [int(args.video_width), int(args.video_height)],
                "camera": observer_camera_model.name,
                "source": "observer-camera",
                "used_for_vision": False,
                "strict": bool(args.recording_strict),
            }
        write_json_result(out, result)
        print(json.dumps(result, indent=2), flush=True)
        print(
            f"[V7 BENCHMARK] completed {result['chain_successes']}/{len(validation_envs)}",
            flush=True,
        )
    except Exception:
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
        app.close()


if __name__ == "__main__":
    main()
