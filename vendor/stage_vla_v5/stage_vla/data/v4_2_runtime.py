"""Isaac Lab runtime helpers for the v4.2 expert-data pilot.

Keep this module separate from :mod:`v4_2_contract` so the critical dataset
contract remains importable/testable on machines without Isaac Lab.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stage_vla.core.project_config import read_float, read_int
from stage_vla.data.v4_2_contract import STATE_DIM
from stage_vla.envs.state_readers import read_grasp_state, read_placement_state, to_torch
from stage_vla.envs.strict_success_observer import StrictM7SuccessObserver
from stage_vla.stages import (
    LiftConfig,
    PhysicalGraspConfig,
    RedOnBlueConfig,
    RedOnBlueSuccessConfig,
    StableGraspConfig,
)


def build_strict_observer(base_env, config_file: str | Path) -> StrictM7SuccessObserver:
    """Build the unchanged frozen strict stack-success observer."""
    cfg = Path(config_file)
    return StrictM7SuccessObserver(
        base_env,
        physical_cfg=PhysicalGraspConfig(
            radial_tolerance_m=read_float(cfg, "grasp", "radial_tolerance_m"),
            height_tolerance_m=read_float(cfg, "grasp", "height_tolerance_m"),
            contact_force_threshold_n=read_float(cfg, "grasp", "contact_force_threshold_n"),
            endpoint_margin=read_float(cfg, "grasp", "endpoint_margin"),
        ),
        stable_cfg=StableGraspConfig(
            required_consecutive_steps=read_int(cfg, "stable_grasp", "required_consecutive_steps")
        ),
        lift_cfg=LiftConfig(
            minimum_object_lift_delta_m=read_float(cfg, "lift", "minimum_object_lift_delta_m")
        ),
        placement_cfg=RedOnBlueConfig(
            xy_tolerance_m=read_float(cfg, "red_on_blue_success", "xy_tolerance_m"),
            target_height_diff_m=read_float(cfg, "red_on_blue_success", "target_height_diff_m"),
            height_tolerance_m=read_float(cfg, "red_on_blue_success", "height_tolerance_m"),
            max_red_linear_speed_mps=read_float(cfg, "red_on_blue_success", "max_red_linear_speed_mps"),
            max_red_angular_speed_radps=read_float(cfg, "red_on_blue_success", "max_red_angular_speed_radps"),
            max_blue_linear_speed_mps=read_float(cfg, "red_on_blue_success", "max_blue_linear_speed_mps"),
            max_blue_angular_speed_radps=read_float(cfg, "red_on_blue_success", "max_blue_angular_speed_radps"),
        ),
        success_cfg=RedOnBlueSuccessConfig(
            required_settle_steps=read_int(cfg, "red_on_blue_success", "required_settle_steps")
        ),
        gripper_open_tolerance_m=read_float(cfg, "red_on_blue_success", "gripper_open_tolerance_m"),
    )


def install_dataset_camera(
    env_cfg,
    *,
    width: int = 84,
    height: int = 84,
    update_period_s: float = 0.05,
    camera_name: str = "v4_dataset_camera",
    position: tuple[float, float, float] = (1.0, 0.0, 0.33),
    rotation_wxyz: tuple[float, float, float, float] = (-0.3799, 0.5963, 0.5963, -0.3799),
    focal_length: float = 24.0,
    horizontal_aperture: float = 20.955,
    data_types: tuple[str, ...] = ("rgb",),
) -> None:
    """Add one fixed third-person RGB camera to every cloned environment.

    The camera is parented under ``{ENV_REGEX_NS}``, so its pose is local to each
    cloned environment and does not leak world-space env-origin offsets.
    """
    if width < 32 or height < 32:
        raise ValueError("dataset camera width/height must both be >= 32")
    if update_period_s <= 0:
        raise ValueError("camera update_period_s must be > 0")
    if len(position) != 3 or len(rotation_wxyz) != 4:
        raise ValueError("camera position/rotation must contain 3/4 values")
    if focal_length <= 0 or horizontal_aperture <= 0:
        raise ValueError("camera focal length and aperture must be positive")

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    camera_cfg = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{camera_name}",
        update_period=float(update_period_s),
        height=int(height),
        width=int(width),
        data_types=[str(name) for name in data_types],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(focal_length),
            focus_distance=400.0,
            horizontal_aperture=float(horizontal_aperture),
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in position),
            rot=tuple(float(value) for value in rotation_wxyz),
            convention="ros",
        ),
    )
    setattr(env_cfg.scene, camera_name, camera_cfg)


def read_rgb_u8(base_env, *, camera_name: str = "v4_dataset_camera") -> np.ndarray:
    """Read env-0 RGB as contiguous uint8 ``[H,W,3]``."""
    camera = base_env.scene[camera_name]
    if "rgb" not in camera.data.output:
        raise RuntimeError(f"camera {camera_name!r} has no rgb output")
    rgb = to_torch(camera.data.output["rgb"])
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] < 3:
        raise RuntimeError(f"unexpected camera RGB shape: {tuple(rgb.shape)}")
    rgb = rgb[0, ..., :3]
    if torch.is_floating_point(rgb):
        # Isaac camera RGB is normally uint8, but handle normalized float output
        # explicitly rather than silently casting 0..1 to black.
        max_value = float(rgb.max().item()) if rgb.numel() else 0.0
        if max_value <= 1.0 + 1e-6:
            rgb = rgb * 255.0
        rgb = rgb.round().clamp(0.0, 255.0).to(torch.uint8)
    else:
        rgb = rgb.to(torch.uint8)
    return np.ascontiguousarray(rgb.detach().cpu().numpy())


def read_rgb_u8_batch(base_env, *, camera_name: str = "v4_dataset_camera") -> np.ndarray:
    """Read all environment RGB frames as contiguous uint8 ``[N,H,W,3]``."""
    camera = base_env.scene[camera_name]
    if "rgb" not in camera.data.output:
        raise RuntimeError(f"camera {camera_name!r} has no rgb output")
    rgb = to_torch(camera.data.output["rgb"])
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise RuntimeError(f"unexpected batched camera RGB shape: {tuple(rgb.shape)}")
    rgb = rgb[..., :3]
    if torch.is_floating_point(rgb):
        max_value = float(rgb.max().item()) if rgb.numel() else 0.0
        if max_value <= 1.0 + 1e-6:
            rgb = rgb * 255.0
        rgb = rgb.round().clamp(0.0, 255.0).to(torch.uint8)
    else:
        rgb = rgb.to(torch.uint8)
    return np.ascontiguousarray(rgb.detach().cpu().numpy())


def read_depth_m(base_env, *, camera_name: str = "v4_dataset_camera") -> np.ndarray:
    """Read env-0 ``distance_to_image_plane`` as float32 metres."""
    camera = base_env.scene[camera_name]
    key = "distance_to_image_plane"
    if key not in camera.data.output:
        raise RuntimeError(f"camera {camera_name!r} has no {key} output")
    depth = to_torch(camera.data.output[key])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[0, ..., 0]
    elif depth.ndim == 3:
        depth = depth[0]
    else:
        raise RuntimeError(f"unexpected camera depth shape: {tuple(depth.shape)}")
    return np.ascontiguousarray(depth.detach().cpu().numpy().astype(np.float32, copy=False))


def read_depth_m_batch(base_env, *, camera_name: str = "v4_dataset_camera") -> np.ndarray:
    """Read all environment depth frames as float32 metres ``[N,H,W]``."""
    camera = base_env.scene[camera_name]
    key = "distance_to_image_plane"
    if key not in camera.data.output:
        raise RuntimeError(f"camera {camera_name!r} has no {key} output")
    depth = to_torch(camera.data.output[key])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim != 3:
        raise RuntimeError(f"unexpected batched camera depth shape: {tuple(depth.shape)}")
    return np.ascontiguousarray(depth.detach().cpu().numpy().astype(np.float32, copy=False))


def read_state_vector(base_env) -> np.ndarray:
    """Read the frozen 57-D v4.2 state vector from env 0.

    All *positions* are converted to environment-local coordinates.  Quaternions
    and velocities keep their native Isaac world orientation/velocity convention;
    they do not contain the clone origin translation that caused the V3 D9 leak.
    """
    if int(base_env.num_envs) != 1:
        raise ValueError("v4.2 pilot capture requires exactly one environment")

    scene = base_env.scene
    origin = to_torch(scene.env_origins)[0, :3]
    robot = scene["robot"]
    joint_pos = to_torch(robot.data.joint_pos)[0].reshape(-1)
    joint_vel = to_torch(robot.data.joint_vel)[0].reshape(-1)
    if joint_pos.numel() != 9 or joint_vel.numel() != 9:
        raise RuntimeError(
            f"expected Franka 9 joint positions/velocities, got {joint_pos.numel()}/{joint_vel.numel()}"
        )

    grasp = read_grasp_state(base_env)
    placement = read_placement_state(base_env, gripper_open_tolerance_m=0.002)
    red = scene["cube_2"]
    blue = scene["cube_1"]
    red_quat = to_torch(red.data.root_quat_w)[0, :4]
    blue_quat = to_torch(blue.data.root_quat_w)[0, :4]

    values = torch.cat(
        [
            joint_pos,
            joint_vel,
            grasp.ee_pos_w[0] - origin,
            grasp.left_tip_w[0] - origin,
            grasp.right_tip_w[0] - origin,
            placement.red_pos_w[0] - origin,
            red_quat,
            placement.red_lin_vel_w[0],
            placement.red_ang_vel_w[0],
            placement.blue_pos_w[0] - origin,
            blue_quat,
            placement.blue_lin_vel_w[0],
            placement.blue_ang_vel_w[0],
            placement.gripper_joint_pos[0],
            grasp.finger_a_force_n[0].reshape(1),
            grasp.finger_b_force_n[0].reshape(1),
        ],
        dim=0,
    ).to(dtype=torch.float32)
    if values.numel() != STATE_DIM:
        raise RuntimeError(f"v4.2 state vector has {values.numel()} values, expected {STATE_DIM}")
    if not torch.isfinite(values).all():
        raise RuntimeError("v4.2 state vector contains NaN/Inf")
    return values.detach().cpu().numpy().astype(np.float32, copy=False)
