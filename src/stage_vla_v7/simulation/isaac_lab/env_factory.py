"""V7-owned Isaac Lab environment factory with explicit legacy adapters."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

from ..config import CameraSpec, validate_camera_specs


class SimulationEnvironmentFactory:
    """Fail-closed factory for physical Simulation environments."""

    supported_environments = ("red_on_blue",)

    def create(self, environment: str, **kwargs):
        if environment not in self.supported_environments:
            raise LookupError(f"unknown Simulation environment {environment!r}")
        return make_known_size_grasp_env(**kwargs)


def create_environment(environment: str, **kwargs):
    return SimulationEnvironmentFactory().create(environment, **kwargs)


def make_known_size_grasp_env(
    *,
    device,
    num_envs,
    seed,
    log_dir,
    known_size,
    effort_limit: float = 40.0,
    arm_scale: float = 1.0,
    camera_name: str | None = None,
    camera_arm_scale: float = 0.1,
    preserve_oracle_physics_with_camera: bool = False,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_position=(1.0, 0.0, 0.40),
    camera_rotation_wxyz=(-0.61237, -0.61237, 0.35355, 0.35355),
    camera_focal_length: float = 24.0,
    camera_horizontal_aperture: float = 20.955,
    camera_data_types=("rgb",),
    camera_specs: tuple[CameraSpec, ...] | None = None,
    fixed_blue_xyz=None,
    fixed_red_xyz=None,
    fixed_asset_xyz=None,
    use_known_size_gripper: bool = True,
    disable_terminations: bool = True,
    object_geometry: str = "box",
    object_color_rgb=(220, 40, 40),
    extra_cube_colors: Mapping[str, tuple[int, int, int]] | None = None,
    physical_batch=None,
    fixed_tilt: bool = False,
):
    """Create an Isaac stack scene with continuous size-conditioned gripper control.

    ``known_size`` is a :class:`KnownSizeGraspConfig`.  The factory keeps the
    arm's relative-pose IK action and replaces only the binary gripper action
    term.  All simulator termination terms are disabled so the training
    wrapper can enforce its own finite-horizon contract.
    """
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from stage_vla.envs import install_finger_net_contact_sensors
    from stage_vla.envs.known_size_grasp_action import KnownSizeGraspAction, KnownSizeGraspActionCfg
    from stage_vla.envs.state_readers import to_torch

    from stage_vla.rl.known_size_grasp import KnownSizeGraspConfig
    from stage_vla.rl.object_physics import PhysicalObjectBatch

    if not isinstance(known_size, KnownSizeGraspConfig):
        raise TypeError("known_size must be KnownSizeGraspConfig")
    known_size.validate()
    if physical_batch is not None:
        if not isinstance(physical_batch, PhysicalObjectBatch):
            raise TypeError("physical_batch must be a PhysicalObjectBatch")
        if physical_batch.num_envs != int(num_envs):
            raise ValueError("physical_batch size must equal num_envs")
    if not 0 < float(effort_limit) <= 200:
        raise ValueError("effort_limit must be in (0, 200] N")
    if not 0 < float(arm_scale) <= 1:
        raise ValueError("arm_scale must be in (0, 1]")
    if not 0 < float(camera_arm_scale) <= 1:
        raise ValueError("camera_arm_scale must be in (0, 1]")
    configured_cameras = tuple(camera_specs or ())
    if camera_name is not None:
        if configured_cameras:
            raise ValueError("camera_name and camera_specs cannot be combined")
        configured_cameras = (
            CameraSpec(
                name=str(camera_name),
                role=(
                    "vision"
                    if "distance_to_image_plane" in tuple(camera_data_types)
                    else "observer"
                ),
                width=int(camera_width),
                height=int(camera_height),
                data_types=tuple(str(value) for value in camera_data_types),
                position_m=tuple(float(value) for value in camera_position),
                rotation_wxyz=tuple(float(value) for value in camera_rotation_wxyz),
                focal_length_mm=float(camera_focal_length),
                horizontal_aperture_mm=float(camera_horizontal_aperture),
            ),
        )
    validate_camera_specs(configured_cameras)
    if (fixed_blue_xyz is None) != (fixed_red_xyz is None):
        raise ValueError("fixed_blue_xyz and fixed_red_xyz must be supplied together")
    if fixed_asset_xyz is not None and fixed_blue_xyz is not None:
        raise ValueError("fixed_asset_xyz cannot be combined with fixed blue/red positions")
    if object_geometry not in ("box", "cylinder", "cone"):
        raise ValueError("object_geometry must be box, cylinder or cone")
    color = tuple(float(value) / 255.0 for value in object_color_rgb)
    if len(color) != 3 or any(value < 0.0 or value > 1.0 for value in color):
        raise ValueError("object_color_rgb must contain three values in [0,255]")
    extra_cube_colors = dict(extra_cube_colors or {})
    for name, rgb in extra_cube_colors.items():
        if not name.startswith("cube_") or not name[5:].isdigit() or int(name[5:]) < 4:
            raise ValueError("extra cube names must use cube_N with N >= 4")
        if len(rgb) != 3 or any(not 0 <= int(value) <= 255 for value in rgb):
            raise ValueError(f"extra cube color for {name!r} must contain three values in [0,255]")

    def _pose_param(value):
        rows = list(value)
        if rows and isinstance(rows[0], (list, tuple)):
            return tuple(tuple(float(v) for v in row) for row in rows)
        return tuple(float(v) for v in rows)

    cfg = parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-v0", device=device, num_envs=num_envs)
    cfg.seed = seed
    if fixed_tilt:
        from stage_vla.envs.fixed_tilt_ik import (
            FixedTiltDifferentialInverseKinematicsAction,
        )

        cfg.actions.arm_action.class_type = (
            FixedTiltDifferentialInverseKinematicsAction
        )
    if physical_batch is not None:
        from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
        from stage_vla.envs.physical_profiles import (
            ISAAC_BLOCK_COLLISION_SIZE_M,
            set_rigid_body_scales,
        )

        cfg.scene.replicate_physics = False
        object_base_size = (
            (ISAAC_BLOCK_COLLISION_SIZE_M,) * 3
            if object_geometry == "box"
            else (
                float(known_size.width_m),
                float(known_size.depth_m),
                float(known_size.height_m),
            )
        )
        support_base_size = (ISAAC_BLOCK_COLLISION_SIZE_M,) * 3
        object_scales = (
            physical_batch.object_size_m
            / physical_batch.object_size_m.new_tensor(object_base_size)
        )
        support_scales = (
            physical_batch.support_size_m
            / physical_batch.support_size_m.new_tensor(support_base_size)
        )
        cfg.events.object_physical_scale = EventTerm(
            func=set_rigid_body_scales,
            mode="prestartup",
            params={
                "scales": object_scales.cpu().tolist(),
                "asset_cfg": SceneEntityCfg("cube_2"),
            },
        )
        cfg.events.support_physical_scale = EventTerm(
            func=set_rigid_body_scales,
            mode="prestartup",
            params={
                "scales": support_scales.cpu().tolist(),
                "asset_cfg": SceneEntityCfg("cube_1"),
            },
        )
    if extra_cube_colors:
        import isaaclab.sim as sim_utils

        for offset, (name, rgb) in enumerate(sorted(extra_cube_colors.items()), start=1):
            asset_cfg = deepcopy(cfg.scene.cube_3)
            suffix = int(name[5:])
            asset_cfg.prim_path = f"{{ENV_REGEX_NS}}/Cube_{suffix}"
            asset_cfg.init_state.pos = [0.60, -0.10 + 0.05 * offset, 0.0203]
            asset_cfg.spawn.semantic_tags = [("class", name)]
            asset_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(float(value) / 255.0 for value in rgb)
            )
            setattr(cfg.scene, name, asset_cfg)
        if fixed_asset_xyz is None:
            from isaaclab.managers import SceneEntityCfg

            cfg.events.randomize_cube_positions.params["asset_cfgs"].extend(
                SceneEntityCfg(name) for name in sorted(extra_cube_colors)
            )
    if object_geometry != "box":
        import isaaclab.sim as sim_utils

        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=float(known_size.mass_kg)),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            # Keep the controller's friction model and Isaac contact model in
            # the same object profile.  Without an explicit material the
            # configured coefficient only changes the force target and tapered
            # objects can never carry their own weight during LIFT.
            "physics_material": sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="max",
                static_friction=float(known_size.friction_coefficient),
                dynamic_friction=float(known_size.friction_coefficient),
                restitution=0.0,
            ),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            "semantic_tags": [("class", "cube_2")],
        }
        if object_geometry == "cylinder":
            spawn = sim_utils.CylinderCfg(
                radius=float(known_size.width_m) / 2.0,
                height=float(known_size.height_m),
                **common,
            )
        else:
            spawn = sim_utils.ConeCfg(
                radius=float(known_size.width_m) / 2.0,
                height=float(known_size.height_m),
                **common,
            )
        cfg.scene.cube_2.spawn = spawn
    # Visual policies use a conservative IK scale by default. Recording an
    # oracle policy can explicitly preserve its camera-free action contract.
    cfg.actions.arm_action.scale = float(
        camera_arm_scale if configured_cameras else arm_scale
    )
    cfg.episode_length_s = 1e6
    if disable_terminations:
        cfg.is_finite_horizon = True
    if disable_terminations:
        termination_names = [
            "time_out", "success", "cube_1_dropping", "cube_2_dropping", "cube_3_dropping",
            *(f"{name}_dropping" for name in extra_cube_colors),
        ]
        for name in termination_names:
            if hasattr(cfg.terminations, name):
                setattr(cfg.terminations, name, None)
    if fixed_blue_xyz is not None:
        # Keep the upstream reset sampler (including a randomized cube_3),
        # then apply the requested pair as the final reset event.
        from isaaclab.managers import EventTermCfg as EventTerm
        from stage_vla.envs.fixed_object_pose import set_fixed_pair_pose

        cfg.events.fixed_pair_pose = EventTerm(
            func=set_fixed_pair_pose,
            mode="reset",
            params={
                "blue_xyz": _pose_param(fixed_blue_xyz),
                "red_xyz": _pose_param(fixed_red_xyz),
            },
        )
    elif fixed_asset_xyz is not None:
        from isaaclab.managers import EventTermCfg as EventTerm
        from stage_vla.envs.fixed_object_pose import set_fixed_asset_poses

        cfg.events.fixed_asset_poses = EventTerm(
            func=set_fixed_asset_poses,
            mode="reset",
            params={
                "asset_xyz": {
                    str(name): _pose_param(value)
                    for name, value in fixed_asset_xyz.items()
                },
            },
        )
    preserve_oracle_physics = not configured_cameras or preserve_oracle_physics_with_camera
    if preserve_oracle_physics:
        cfg.sim.physics = deepcopy(getattr(cfg.sim.physics, "default", cfg.sim.physics))
    # Keep the contact-ordering override for camera-free oracle runs.  In
    # Isaac Sim 6.0 the override can terminate the process when a camera is
    # dynamically added to the scene before gym.make().
    if preserve_oracle_physics:
        cfg.sim.physics.solve_articulation_contact_last = True
    if preserve_oracle_physics:
        cfg.scene.robot = deepcopy(cfg.scene.robot)
    cfg.scene.robot.actuators["panda_hand"].effort_limit_sim = float(effort_limit)
    if configured_cameras:
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim_utils

        for camera in configured_cameras:
            if hasattr(cfg.scene, camera.name):
                raise ValueError(f"camera ID conflicts with scene entity: {camera.name!r}")
            setattr(cfg.scene, camera.name, CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{camera.name}",
                update_period=float(camera.update_period_s),
                height=int(camera.height),
                width=int(camera.width),
                data_types=list(camera.data_types),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(camera.focal_length_mm),
                    focus_distance=400.0,
                    horizontal_aperture=float(camera.horizontal_aperture_mm),
                    clipping_range=(0.1, 4.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(v) for v in camera.position_m),
                    rot=tuple(float(v) for v in camera.rotation_wxyz),
                    convention="ros",
                ),
            ))
    install_finger_net_contact_sensors(cfg)
    known_gripper_cfg = KnownSizeGraspActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        width_m=known_size.width_m,
        depth_m=known_size.depth_m,
        height_m=known_size.height_m,
        mass_kg=known_size.mass_kg,
        grasp_width_ratio=(
            1.0
            if known_size.grasp_width_m is None
            else float(known_size.grasp_width_m) / float(known_size.width_m)
        ),
        jaw_clearance_m=known_size.jaw_clearance_m,
        max_compression_m=known_size.max_compression_m,
        friction_coefficient=known_size.friction_coefficient,
        safety_factor=known_size.safety_factor,
        min_force_n=known_size.min_force_n,
        max_force_n=known_size.max_force_n,
        pressure_tolerance_n=known_size.pressure_tolerance_n,
        force_balance_tolerance_n=known_size.force_balance_tolerance_n,
        residual_force_range_n=known_size.residual_force_range_n,
        residual_deadband=0.05 if physical_batch is not None else 0.0,
        lift_acceleration_mps2=known_size.lift_acceleration_mps2,
        joint_min_m=known_size.joint_min_m,
        joint_max_m=known_size.joint_max_m,
        open_position_m=min(float(getattr(cfg, "gripper_open_val", known_size.joint_max_m)), known_size.joint_max_m),
    )
    if use_known_size_gripper:
        cfg.actions.gripper_action = known_gripper_cfg
    cfg.log_dir = str(log_dir)
    raw = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=cfg)
    arm_term = raw.unwrapped.action_manager.get_term("arm_action")
    term = raw.unwrapped.action_manager.get_term("gripper_action")
    if physical_batch is not None:
        from stage_vla.envs.physical_profiles import (
            read_rigid_body_scales,
            set_rigid_body_mass_profile,
            verify_rigid_body_mass_profile,
        )

        expected_object_scales = object_scales.cpu()
        expected_support_scales = support_scales.cpu()
        actual_object_scales = read_rigid_body_scales(raw.unwrapped, "cube_2")
        actual_support_scales = read_rigid_body_scales(raw.unwrapped, "cube_1")
        if not torch.allclose(actual_object_scales, expected_object_scales, atol=1e-5, rtol=1e-5):
            raw.close()
            raise RuntimeError("simulator object scale does not match the physical profile")
        if not torch.allclose(actual_support_scales, expected_support_scales, atol=1e-5, rtol=1e-5):
            raw.close()
            raise RuntimeError("simulator support scale does not match the physical profile")
        set_rigid_body_mass_profile(
            raw.unwrapped.scene["cube_2"], physical_batch.object_mass_kg
        )
        set_rigid_body_mass_profile(
            raw.unwrapped.scene["cube_1"], physical_batch.support_mass_kg
        )
        actual_object_mass = verify_rigid_body_mass_profile(
            raw.unwrapped.scene["cube_2"], physical_batch.object_mass_kg
        )
        actual_support_mass = verify_rigid_body_mass_profile(
            raw.unwrapped.scene["cube_1"], physical_batch.support_mass_kg
        )
        if use_known_size_gripper:
            term.set_physical_batch(physical_batch)
        raw.unwrapped.physical_profile_verification = {
            "block_collision_base_size_m": ISAAC_BLOCK_COLLISION_SIZE_M,
            "object_scale": actual_object_scales,
            "support_scale": actual_support_scales,
            "object_mass_kg": actual_object_mass,
            "support_mass_kg": actual_support_mass,
        }
    robot = raw.unwrapped.scene["robot"]
    ids, _ = robot.find_joints(cfg.gripper_joint_names)
    actual = to_torch(robot.data.joint_effort_limits)[:, ids]
    if (
        (
            fixed_tilt
            and arm_term.__class__.__name__
            != "FixedTiltDifferentialInverseKinematicsAction"
        )
        or
        (use_known_size_gripper and not isinstance(term, KnownSizeGraspAction))
        or len(ids) != 2
        or not ((actual - effort_limit).abs() < 1e-5).all()
        or (disable_terminations and raw.unwrapped.termination_manager.active_terms)
    ):
        raw.close()
        raise RuntimeError("known-size gripper action, effort cap, or termination contract mismatch")
    return raw


def install_known_size_gripper_action(raw, known_size, effort_limit: float = 40.0):
    """Replace the native binary gripper term in-place after visual REACH.

    Keeping the native binary term during camera-driven REACH avoids a
    renderer/feedback interaction in Isaac Sim 6.0.  The replacement preserves
    the action-manager ordering and one-dimensional gripper interface, so the
    v5 pressure-feedback policy can take over without resetting the episode.
    """
    from stage_vla.envs.known_size_grasp_action import KnownSizeGraspAction, KnownSizeGraspActionCfg

    cfg = KnownSizeGraspActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        width_m=known_size.width_m, depth_m=known_size.depth_m,
        height_m=known_size.height_m, mass_kg=known_size.mass_kg,
        grasp_width_ratio=(
            1.0
            if known_size.grasp_width_m is None
            else float(known_size.grasp_width_m) / float(known_size.width_m)
        ),
        jaw_clearance_m=known_size.jaw_clearance_m,
        max_compression_m=known_size.max_compression_m,
        friction_coefficient=known_size.friction_coefficient,
        safety_factor=known_size.safety_factor, min_force_n=known_size.min_force_n,
        max_force_n=known_size.max_force_n,
        pressure_tolerance_n=known_size.pressure_tolerance_n,
        force_balance_tolerance_n=known_size.force_balance_tolerance_n,
        residual_force_range_n=known_size.residual_force_range_n,
        lift_acceleration_mps2=known_size.lift_acceleration_mps2,
        joint_min_m=known_size.joint_min_m, joint_max_m=known_size.joint_max_m,
        open_position_m=min(known_size.joint_max_m, 0.04),
    )
    raw.unwrapped.action_manager.get_term("gripper_action")
    # Reuse the already-created sensor references and action-manager slot.
    term = KnownSizeGraspAction(cfg, raw.unwrapped)
    manager = raw.unwrapped.action_manager
    manager._terms["gripper_action"] = term
    if hasattr(raw.unwrapped.cfg.actions, "gripper_action"):
        raw.unwrapped.cfg.actions.gripper_action = cfg
    term.reset()
    return term
