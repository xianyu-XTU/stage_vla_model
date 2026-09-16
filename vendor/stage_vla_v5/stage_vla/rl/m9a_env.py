"""Build the M9-A state-based continuous-action PPO training variant.

This modifies only the parsed task *configuration before gym.make()*:
- pose-relative Differential IK remains active in the base task (6 arm + gripper = 7 raw actions);
- PPO is exposed only XYZ + gripper through a 4->7 action adapter;
- dRx/dRy are zero while dRz is a deterministic cube-edge alignment correction;
- safe per-axis IK translation scale for normalized PPO actions;
- verified per-finger net-force sensors;
- non-stage-aware baseline reward terms;
- removes the unrelated official three-cube success termination.

No Isaac Lab core code is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stage_vla.core import read_float
from stage_vla.envs import install_finger_net_contact_sensors
from stage_vla.stages import PhysicalGraspConfig, RedOnBlueConfig


@dataclass(frozen=True)
class M9AEnvBuildSummary:
    policy_action_dim_expected: int
    raw_action_dim_expected: int
    arm_command_type: str
    arm_relative: bool
    arm_scale_m_per_axis: float
    rotation_action_scale_rad: float
    alignment_max_yaw_step_rad: float
    alignment_tolerance_deg: float
    reward_terms: tuple[str, ...]


def _load_configs(config_file: Path):
    physical = PhysicalGraspConfig(
        radial_tolerance_m=read_float(config_file, "grasp", "radial_tolerance_m"),
        height_tolerance_m=read_float(config_file, "grasp", "height_tolerance_m"),
        contact_force_threshold_n=read_float(config_file, "grasp", "contact_force_threshold_n"),
        endpoint_margin=read_float(config_file, "grasp", "endpoint_margin"),
    )
    placement = RedOnBlueConfig(
        xy_tolerance_m=read_float(config_file, "red_on_blue_success", "xy_tolerance_m"),
        target_height_diff_m=read_float(config_file, "red_on_blue_success", "target_height_diff_m"),
        height_tolerance_m=read_float(config_file, "red_on_blue_success", "height_tolerance_m"),
        max_red_linear_speed_mps=read_float(config_file, "red_on_blue_success", "max_red_linear_speed_mps"),
        max_red_angular_speed_radps=read_float(config_file, "red_on_blue_success", "max_red_angular_speed_radps"),
        max_blue_linear_speed_mps=read_float(config_file, "red_on_blue_success", "max_blue_linear_speed_mps"),
        max_blue_angular_speed_radps=read_float(config_file, "red_on_blue_success", "max_blue_angular_speed_radps"),
    )
    return physical, placement


def _physical_grasp_reward_params(cfg: PhysicalGraspConfig) -> dict[str, float]:
    """Return configclass-safe primitive params for the M9-A grasp term."""
    return {
        "radial_tolerance_m": float(cfg.radial_tolerance_m),
        "height_tolerance_m": float(cfg.height_tolerance_m),
        "contact_force_threshold_n": float(cfg.contact_force_threshold_n),
        "endpoint_margin": float(cfg.endpoint_margin),
    }


def _current_success_reward_params(
    physical_cfg: PhysicalGraspConfig,
    placement_cfg: RedOnBlueConfig,
    *,
    gripper_open_tolerance_m: float,
) -> dict[str, float]:
    """Return primitive-only params for the stateless current-success term."""
    return {
        "xy_tolerance_m": float(placement_cfg.xy_tolerance_m),
        "target_height_diff_m": float(placement_cfg.target_height_diff_m),
        "height_tolerance_m": float(placement_cfg.height_tolerance_m),
        "max_red_linear_speed_mps": float(placement_cfg.max_red_linear_speed_mps),
        "max_red_angular_speed_radps": float(placement_cfg.max_red_angular_speed_radps),
        "max_blue_linear_speed_mps": float(placement_cfg.max_blue_linear_speed_mps),
        "max_blue_angular_speed_radps": float(placement_cfg.max_blue_angular_speed_radps),
        "radial_tolerance_m": float(physical_cfg.radial_tolerance_m),
        "grasp_height_tolerance_m": float(physical_cfg.height_tolerance_m),
        "contact_force_threshold_n": float(physical_cfg.contact_force_threshold_n),
        "endpoint_margin": float(physical_cfg.endpoint_margin),
        "gripper_open_tolerance_m": float(gripper_open_tolerance_m),
    }


def configure_m9a_baseline_env(env_cfg, *, config_file: Path) -> M9AEnvBuildSummary:
    """Mutate a freshly parsed stack env cfg into the M9-A training variant."""
    from isaaclab.envs import mdp as generic_mdp
    from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

    from stage_vla.rl import isaac_reward_terms as terms

    physical_cfg, placement_cfg = _load_configs(config_file)

    # M9-A and M9-B must expose the same *learned* low-level DOFs: XYZ +
    # gripper.  Earlier v0.9.3 changed the controller itself to position-only
    # IK and incorrectly assumed that this held EE orientation.  Isaac Lab's
    # DifferentialIKController uses only the 3xN position Jacobian in that
    # mode, so orientation is not controlled.  Keep pose-relative IK and let
    # the RSL adapter expand [dx,dy,dz,grip] -> [dx,dy,dz,0,0,0,grip].
    arm = env_cfg.actions.arm_action
    arm.controller.command_type = "pose"
    arm.controller.use_relative_mode = True
    action_scale = read_float(config_file, "m9a_ppo", "arm_translation_scale_m")
    if action_scale <= 0:
        raise ValueError("m9a_ppo.arm_translation_scale_m must be > 0")
    # Translation is learned and capped at the same +/-4 mm scale used by the
    # previous baseline. Roll/pitch remain deterministic zero. Yaw is supplied
    # by the shared edge-alignment controller (not by PPO), so keep an explicit
    # action scale for converting processed yaw radians to the raw dRz channel.
    rotation_action_scale = read_float(config_file, "m9a_grasp_alignment", "rotation_action_scale_rad")
    alignment_max_yaw_step = read_float(config_file, "m9a_grasp_alignment", "max_yaw_step_rad")
    alignment_tolerance_deg = read_float(config_file, "m9a_grasp_alignment", "tolerance_deg")
    if rotation_action_scale <= 0:
        raise ValueError("m9a_grasp_alignment.rotation_action_scale_rad must be > 0")
    if alignment_max_yaw_step <= 0:
        raise ValueError("m9a_grasp_alignment.max_yaw_step_rad must be > 0")
    if not (0.0 <= alignment_tolerance_deg < 45.0):
        raise ValueError("m9a_grasp_alignment.tolerance_deg must satisfy 0 <= value < 45")
    arm.scale = (
        action_scale,
        action_scale,
        action_scale,
        rotation_action_scale,
        rotation_action_scale,
        rotation_action_scale,
    )

    install_finger_net_contact_sensors(env_cfg)

    resting_z = read_float(config_file, "m9a_reward", "resting_cube_center_z_m")
    lift_target = read_float(config_file, "m9a_reward", "lift_target_delta_m")
    goal_gate = read_float(config_file, "m9a_reward", "goal_gate_lift_delta_m")
    reach_std = read_float(config_file, "m9a_reward", "reach_std_m")
    pregrasp_tip_height_offset = read_float(config_file, "m9a_reward", "pregrasp_tip_height_offset_m")
    pregrasp_xy_std = read_float(config_file, "m9a_reward", "pregrasp_xy_std_m")
    pregrasp_z_std = read_float(config_file, "m9a_reward", "pregrasp_z_std_m")
    gripper_closed_joint_pos = read_float(config_file, "m9a_reward", "gripper_closed_joint_pos_m")
    premature_close_penalty_scale = read_float(
        config_file, "m9a_reward", "premature_close_penalty_scale"
    )
    goal_std = read_float(config_file, "m9a_reward", "goal_std_m")
    goal_fine_std = read_float(config_file, "m9a_reward", "goal_fine_std_m")

    # M9-A4: current-frame post-lift corridor + release timing.  These remain
    # conventional baseline terms: no stage tracker/history/potential state.
    cube_size = read_float(config_file, "task", "cube_size_m")
    transport_surface_clearance = read_float(
        config_file, "red_on_blue_success", "transport_surface_clearance_m"
    )
    postgrasp_blend_radius = read_float(config_file, "m9a_reward", "postgrasp_blend_xy_radius_m")
    postgrasp_xy_std = read_float(config_file, "m9a_reward", "postgrasp_xy_std_m")
    postgrasp_z_std = read_float(config_file, "m9a_reward", "postgrasp_z_std_m")
    release_xy_tol = read_float(config_file, "m9a_reward", "release_xy_tolerance_m")
    release_height_tol = read_float(config_file, "m9a_reward", "release_height_tolerance_m")
    premature_open_penalty = read_float(
        config_file, "m9a_reward", "premature_open_penalty_scale"
    )
    gripper_tol = read_float(config_file, "red_on_blue_success", "gripper_open_tolerance_m")

    rewards = {
        "reach_red": RewTerm(
            func=terms.reach_red_reward,
            weight=read_float(config_file, "m9a_reward", "reach_weight"),
            params={"std_m": reach_std},
        ),
        "pregrasp_pose": RewTerm(
            func=terms.pregrasp_pose_reward,
            weight=read_float(config_file, "m9a_reward", "pregrasp_pose_weight"),
            params={
                "target_tip_height_offset_m": pregrasp_tip_height_offset,
                "xy_std_m": pregrasp_xy_std,
                "z_std_m": pregrasp_z_std,
            },
        ),
        "gripper_coordination": RewTerm(
            func=terms.gripper_coordination_reward,
            weight=read_float(config_file, "m9a_reward", "gripper_coordination_weight"),
            params={
                "closed_joint_pos_m": gripper_closed_joint_pos,
                "premature_close_penalty_scale": premature_close_penalty_scale,
                **_physical_grasp_reward_params(physical_cfg),
            },
        ),
        "physical_grasp": RewTerm(
            func=terms.physical_grasp_reward,
            weight=read_float(config_file, "m9a_reward", "grasp_weight"),
            # IMPORTANT: RewardTermCfg is an Isaac Lab configclass.  In the
            # user's Isaac Lab 3.0 runtime its post-init recursively processes
            # params and tries to assign into nested dataclasses.  Passing our
            # frozen PhysicalGraspConfig therefore raises FrozenInstanceError.
            # Keep manager params configclass-safe: scalars only.
            params=_physical_grasp_reward_params(physical_cfg),
        ),
        "lift_progress": RewTerm(
            func=terms.red_lift_progress_reward,
            weight=read_float(config_file, "m9a_reward", "lift_weight"),
            params={
                "resting_center_z_m": resting_z,
                "target_lift_delta_m": lift_target,
                **_physical_grasp_reward_params(physical_cfg),
            },
        ),
        "goal_tracking": RewTerm(
            func=terms.red_goal_tracking_reward,
            weight=read_float(config_file, "m9a_reward", "goal_weight"),
            params={
                "resting_center_z_m": resting_z,
                "gate_lift_delta_m": goal_gate,
                "target_height_diff_m": placement_cfg.target_height_diff_m,
                "std_m": goal_std,
                "gripper_open_tolerance_m": gripper_tol,
                **_physical_grasp_reward_params(physical_cfg),
            },
        ),
        "goal_tracking_fine": RewTerm(
            func=terms.red_goal_tracking_reward,
            weight=read_float(config_file, "m9a_reward", "goal_fine_weight"),
            params={
                "resting_center_z_m": resting_z,
                "gate_lift_delta_m": goal_gate,
                "target_height_diff_m": placement_cfg.target_height_diff_m,
                "std_m": goal_fine_std,
                "gripper_open_tolerance_m": gripper_tol,
                **_physical_grasp_reward_params(physical_cfg),
            },
        ),
        "postgrasp_pose": RewTerm(
            func=terms.red_postgrasp_pose_reward,
            weight=read_float(config_file, "m9a_reward", "postgrasp_pose_weight"),
            params={
                "resting_center_z_m": resting_z,
                "gate_lift_delta_m": goal_gate,
                "target_height_diff_m": placement_cfg.target_height_diff_m,
                "cube_height_m": cube_size,
                "transport_surface_clearance_m": transport_surface_clearance,
                "blend_xy_radius_m": postgrasp_blend_radius,
                "xy_std_m": postgrasp_xy_std,
                "z_std_m": postgrasp_z_std,
                "gripper_open_tolerance_m": gripper_tol,
                **_physical_grasp_reward_params(physical_cfg),
            },
        ),
        "release_coordination": RewTerm(
            func=terms.red_release_coordination_reward,
            weight=read_float(config_file, "m9a_reward", "release_coordination_weight"),
            params={
                "resting_center_z_m": resting_z,
                "gate_lift_delta_m": goal_gate,
                "target_height_diff_m": placement_cfg.target_height_diff_m,
                "release_xy_tolerance_m": release_xy_tol,
                "release_height_tolerance_m": release_height_tol,
                "closed_joint_pos_m": gripper_closed_joint_pos,
                "premature_open_penalty_scale": premature_open_penalty,
                "gripper_open_tolerance_m": gripper_tol,
                **_physical_grasp_reward_params(physical_cfg),
            },
        ),
        "current_success": RewTerm(
            func=terms.red_on_blue_current_success_reward,
            weight=read_float(config_file, "m9a_reward", "success_weight"),
            # Same rule as physical_grasp above: do not place project frozen
            # dataclasses inside RewardTermCfg.params.  Reconstruct them inside
            # the stateless reward function from primitive scalars.
            params=_current_success_reward_params(
                physical_cfg,
                placement_cfg,
                gripper_open_tolerance_m=gripper_tol,
            ),
        ),
        "placed_settle": RewTerm(
            func=terms.red_placed_settle_reward,
            weight=read_float(config_file, "m9a_reward", "placed_settle_weight"),
            params={
                "target_height_diff_m": placement_cfg.target_height_diff_m,
                "xy_std_m": read_float(config_file, "m9a_reward", "placed_settle_xy_std_m"),
                "z_std_m": read_float(config_file, "m9a_reward", "placed_settle_z_std_m"),
                "vel_std_m": read_float(config_file, "m9a_reward", "placed_settle_vel_std_mps"),
                "gripper_open_tolerance_m": gripper_tol,
            },
        ),
        "action_rate": RewTerm(
            func=generic_mdp.action_rate_l2,
            weight=read_float(config_file, "m9a_reward", "action_rate_weight"),
        ),
        "joint_vel": RewTerm(
            func=generic_mdp.joint_vel_l2,
            weight=read_float(config_file, "m9a_reward", "joint_vel_weight"),
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }
    env_cfg.rewards = rewards

    # The official Stack task's success means the full three-cube tower.  It is
    # unrelated to this project's red-on-blue baseline and would reset episodes
    # for the wrong objective. Green-cube dropping is likewise irrelevant.
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "cube_3_dropping"):
        env_cfg.terminations.cube_3_dropping = None

    return M9AEnvBuildSummary(
        policy_action_dim_expected=4,
        raw_action_dim_expected=7,
        arm_command_type=str(arm.controller.command_type),
        arm_relative=bool(arm.controller.use_relative_mode),
        arm_scale_m_per_axis=float(action_scale),
        rotation_action_scale_rad=float(rotation_action_scale),
        alignment_max_yaw_step_rad=float(alignment_max_yaw_step),
        alignment_tolerance_deg=float(alignment_tolerance_deg),
        reward_terms=tuple(rewards.keys()),
    )
