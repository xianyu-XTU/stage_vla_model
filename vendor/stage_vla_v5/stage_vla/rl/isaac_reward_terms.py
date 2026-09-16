"""Isaac-runtime reward terms for the M9-A continuous-action baseline.

These terms are intentionally stateless and non-stage-aware.  They are loaded
only inside Isaac Lab and are kept separate from the dependency-free reward
math in :mod:`stage_vla.rl.baseline_reward`.
"""

from __future__ import annotations

import torch

from stage_vla.envs.state_readers import read_grasp_state, read_gripper_joint_positions, read_placement_state
from stage_vla.stages import PhysicalGraspConfig, RedOnBlueConfig, physical_grasp_diagnostics, red_on_blue_diagnostics

from .baseline_reward import (
    current_success_reward,
    gated_goal_tracking_reward,
    gate_reward_by_bool,
    gripper_closedness,
    gripper_coordination_reward as pure_gripper_coordination_reward,
    lift_progress_reward,
    pregrasp_pose_reward as pure_pregrasp_pose_reward,
    postgrasp_pose_reward as pure_postgrasp_pose_reward,
    placement_ready_mask,
    release_coordination_reward as pure_release_coordination_reward,
    smooth_distance_reward,
)


def reach_red_reward(env, *, std_m: float):
    state = read_grasp_state(env)
    distance = torch.linalg.vector_norm(state.ee_pos_w - state.red_pos_w, dim=-1)
    return smooth_distance_reward(distance, std_m=std_m)



def pregrasp_pose_reward(
    env,
    *,
    target_tip_height_offset_m: float,
    xy_std_m: float,
    z_std_m: float,
):
    """Precise current-frame pre-grasp pose shaping.

    Unlike the coarse EE-to-red reach term, this uses the true fingertip
    midpoint and a validated +Z grasp-height offset. No stage/history state is
    involved.
    """
    state = read_grasp_state(env)
    return pure_pregrasp_pose_reward(
        state.red_pos_w,
        state.left_tip_w,
        state.right_tip_w,
        target_tip_height_offset_m=float(target_tip_height_offset_m),
        xy_std_m=float(xy_std_m),
        z_std_m=float(z_std_m),
    )


def gripper_coordination_reward(
    env,
    *,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
    closed_joint_pos_m: float,
    premature_close_penalty_scale: float,
):
    """Current-frame close-timing shaping without stage/history state.

    The gripper receives positive closing reward only when the M4 geometric
    preconditions are already satisfied (red between true fingertips and both
    fingertip heights aligned). Closing before that point receives a small
    penalty. Actual gripper joint positions are used instead of the command.
    """
    state = read_grasp_state(env)
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    diag = physical_grasp_diagnostics(
        state.red_pos_w,
        state.left_tip_w,
        state.right_tip_w,
        state.finger_a_force_n,
        state.finger_b_force_n,
        cfg=physical_cfg,
    )
    geometry_ready = (
        diag.between_fingertips
        & diag.left_height_aligned
        & diag.right_height_aligned
    )

    if not hasattr(env.cfg, "gripper_open_val"):
        raise RuntimeError("Environment cfg has no gripper_open_val.")
    joints = read_gripper_joint_positions(env)
    closedness = gripper_closedness(
        joints,
        open_joint_pos_m=float(env.cfg.gripper_open_val),
        closed_joint_pos_m=float(closed_joint_pos_m),
    )
    return pure_gripper_coordination_reward(
        closedness,
        geometry_ready,
        premature_close_penalty_scale=float(premature_close_penalty_scale),
    )

def physical_grasp_reward(
    env,
    *,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
):
    # RewardTermCfg.params must remain configclass-safe.  Reconstruct the
    # project's immutable config locally rather than passing a frozen dataclass
    # through Isaac Lab's configuration machinery.
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    state = read_grasp_state(env)
    diag = physical_grasp_diagnostics(
        state.red_pos_w,
        state.left_tip_w,
        state.right_tip_w,
        state.finger_a_force_n,
        state.finger_b_force_n,
        cfg=physical_cfg,
    )
    return diag.physical_grasp.to(torch.float32)


def red_lift_progress_reward(
    env,
    *,
    resting_center_z_m: float,
    target_lift_delta_m: float,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
):
    """Current-state lift progress gated by a current physical grasp.

    This remains a non-stage-aware M9-A baseline term: the gate uses only the
    current M4 detector, with no stable-grasp streak, history latch, or stage.
    The gate prevents a policy from earning lift reward by merely knocking or
    bouncing the red cube upward.
    """
    state = read_grasp_state(env)
    progress = lift_progress_reward(
        state.red_pos_w[:, 2],
        resting_center_z_m=resting_center_z_m,
        target_lift_delta_m=target_lift_delta_m,
    )
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    physical = physical_grasp_diagnostics(
        state.red_pos_w,
        state.left_tip_w,
        state.right_tip_w,
        state.finger_a_force_n,
        state.finger_b_force_n,
        cfg=physical_cfg,
    ).physical_grasp
    return gate_reward_by_bool(progress, physical)


def red_goal_tracking_reward(
    env,
    *,
    resting_center_z_m: float,
    gate_lift_delta_m: float,
    target_height_diff_m: float,
    std_m: float,
    gripper_open_tolerance_m: float,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
):
    """Goal tracking gated by *current* physical grasp.

    The existing height gate remains, but height alone can be satisfied by a
    collision/toss.  Requiring the M4 current-frame grasp preserves a causal
    manipulation signal without introducing M8 stage/history state.  After
    release, the separate current_success term provides the terminal signal.
    """
    pstate = read_placement_state(env, gripper_open_tolerance_m=gripper_open_tolerance_m)
    dense = gated_goal_tracking_reward(
        pstate.red_pos_w,
        pstate.blue_pos_w,
        resting_center_z_m=resting_center_z_m,
        gate_lift_delta_m=gate_lift_delta_m,
        target_height_diff_m=target_height_diff_m,
        std_m=std_m,
    )
    gstate = read_grasp_state(env)
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    physical = physical_grasp_diagnostics(
        gstate.red_pos_w,
        gstate.left_tip_w,
        gstate.right_tip_w,
        gstate.finger_a_force_n,
        gstate.finger_b_force_n,
        cfg=physical_cfg,
    ).physical_grasp
    return gate_reward_by_bool(dense, physical)



def red_postgrasp_pose_reward(
    env,
    *,
    resting_center_z_m: float,
    gate_lift_delta_m: float,
    target_height_diff_m: float,
    cube_height_m: float,
    transport_surface_clearance_m: float,
    blend_xy_radius_m: float,
    xy_std_m: float,
    z_std_m: float,
    gripper_open_tolerance_m: float,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
):
    """M9-A4 post-lift transport/placement corridor reward.

    The pure geometric reward is current-frame and then gated by the current M4
    physical grasp.  No M5 streak, TaskHistory, M8 stage, or previous-potential
    state is introduced.
    """
    pstate = read_placement_state(env, gripper_open_tolerance_m=gripper_open_tolerance_m)
    dense = pure_postgrasp_pose_reward(
        pstate.red_pos_w,
        pstate.blue_pos_w,
        resting_center_z_m=float(resting_center_z_m),
        gate_lift_delta_m=float(gate_lift_delta_m),
        target_height_diff_m=float(target_height_diff_m),
        cube_height_m=float(cube_height_m),
        transport_surface_clearance_m=float(transport_surface_clearance_m),
        blend_xy_radius_m=float(blend_xy_radius_m),
        xy_std_m=float(xy_std_m),
        z_std_m=float(z_std_m),
    )
    gstate = read_grasp_state(env)
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    physical = physical_grasp_diagnostics(
        gstate.red_pos_w,
        gstate.left_tip_w,
        gstate.right_tip_w,
        gstate.finger_a_force_n,
        gstate.finger_b_force_n,
        cfg=physical_cfg,
    ).physical_grasp
    return gate_reward_by_bool(dense, physical)


def red_release_coordination_reward(
    env,
    *,
    resting_center_z_m: float,
    gate_lift_delta_m: float,
    target_height_diff_m: float,
    release_xy_tolerance_m: float,
    release_height_tolerance_m: float,
    closed_joint_pos_m: float,
    premature_open_penalty_scale: float,
    gripper_open_tolerance_m: float,
    radial_tolerance_m: float,
    height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
):
    """M9-A4 current-frame release timing reward using actual gripper joints."""
    pstate = read_placement_state(env, gripper_open_tolerance_m=gripper_open_tolerance_m)
    lifted = (
        pstate.red_pos_w[..., 2] - float(resting_center_z_m)
    ) >= float(gate_lift_delta_m)
    ready = placement_ready_mask(
        pstate.red_pos_w,
        pstate.blue_pos_w,
        target_height_diff_m=float(target_height_diff_m),
        xy_tolerance_m=float(release_xy_tolerance_m),
        height_tolerance_m=float(release_height_tolerance_m),
    ) & lifted

    if not hasattr(env.cfg, "gripper_open_val"):
        raise RuntimeError("Environment cfg has no gripper_open_val.")
    closedness = gripper_closedness(
        pstate.gripper_joint_pos,
        open_joint_pos_m=float(env.cfg.gripper_open_val),
        closed_joint_pos_m=float(closed_joint_pos_m),
    )
    openness = 1.0 - closedness

    gstate = read_grasp_state(env)
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    physical = physical_grasp_diagnostics(
        gstate.red_pos_w,
        gstate.left_tip_w,
        gstate.right_tip_w,
        gstate.finger_a_force_n,
        gstate.finger_b_force_n,
        cfg=physical_cfg,
    ).physical_grasp
    carrying = physical & lifted
    return pure_release_coordination_reward(
        openness,
        ready,
        carrying,
        premature_open_penalty_scale=float(premature_open_penalty_scale),
    )

def red_on_blue_current_success_reward(
    env,
    *,
    xy_tolerance_m: float,
    target_height_diff_m: float,
    height_tolerance_m: float,
    max_red_linear_speed_mps: float,
    max_red_angular_speed_radps: float,
    max_blue_linear_speed_mps: float,
    max_blue_angular_speed_radps: float,
    radial_tolerance_m: float,
    grasp_height_tolerance_m: float,
    contact_force_threshold_n: float,
    endpoint_margin: float,
    gripper_open_tolerance_m: float,
):
    # Keep RewardTermCfg.params primitive-only; rebuild immutable project config
    # objects inside the term after Isaac Lab configclass construction is done.
    placement_cfg = RedOnBlueConfig(
        xy_tolerance_m=float(xy_tolerance_m),
        target_height_diff_m=float(target_height_diff_m),
        height_tolerance_m=float(height_tolerance_m),
        max_red_linear_speed_mps=float(max_red_linear_speed_mps),
        max_red_angular_speed_radps=float(max_red_angular_speed_radps),
        max_blue_linear_speed_mps=float(max_blue_linear_speed_mps),
        max_blue_angular_speed_radps=float(max_blue_angular_speed_radps),
    )
    physical_cfg = PhysicalGraspConfig(
        radial_tolerance_m=float(radial_tolerance_m),
        height_tolerance_m=float(grasp_height_tolerance_m),
        contact_force_threshold_n=float(contact_force_threshold_n),
        endpoint_margin=float(endpoint_margin),
    )
    pstate = read_placement_state(env, gripper_open_tolerance_m=gripper_open_tolerance_m)
    placement = red_on_blue_diagnostics(
        pstate.red_pos_w,
        pstate.blue_pos_w,
        pstate.red_lin_vel_w,
        pstate.red_ang_vel_w,
        pstate.blue_lin_vel_w,
        pstate.blue_ang_vel_w,
        cfg=placement_cfg,
    )
    gstate = read_grasp_state(env)
    grasp = physical_grasp_diagnostics(
        gstate.red_pos_w,
        gstate.left_tip_w,
        gstate.right_tip_w,
        gstate.finger_a_force_n,
        gstate.finger_b_force_n,
        cfg=physical_cfg,
    )
    return current_success_reward(
        geometry_ok=placement.geometry_ok,
        settled=placement.settled,
        gripper_open=pstate.gripper_open,
        physical_grasp=grasp.physical_grasp,
    ).to(device=env.device)


def red_descend_to_stack_reward(
    env,
    *,
    target_height_diff_m: float,
    xy_std_m: float,
    z_std_m: float,
    gripper_open_tolerance_m: float,
):
    """Un-gated dense descend-to-stack reward (StARe PLACE).

    Unlike goal_tracking/postgrasp_pose (gated on lift/physical-grasp), this term
    rewards the red cube being at the exact stack pose (XY over blue, Z at stack
    height) regardless of grasp/lift state. This gives the PLACE policy a
    continuous gradient to descend and align even after it drops below the lift
    gate, which the gated terms switch off.
    """
    pstate = read_placement_state(env, gripper_open_tolerance_m=gripper_open_tolerance_m)
    xy_error = torch.linalg.vector_norm(
        pstate.red_pos_w[..., :2] - pstate.blue_pos_w[..., :2], dim=-1
    )
    z_error = torch.abs(
        pstate.red_pos_w[..., 2] - (pstate.blue_pos_w[..., 2] + float(target_height_diff_m))
    )
    xy_score = smooth_distance_reward(xy_error, std_m=float(xy_std_m))
    z_score = smooth_distance_reward(z_error, std_m=float(z_std_m))
    return (xy_score * z_score).to(device=env.device)


def red_placed_settle_reward(
    env,
    *,
    target_height_diff_m: float,
    xy_std_m: float,
    z_std_m: float,
    vel_std_m: float,
    gripper_open_tolerance_m: float,
):
    """Dense placement + settle shaping (M9-A5).

    Rewards the red cube being close to the exact stack pose (XY over blue, Z at
    stack height) AND at rest. Unlike the terminal success term (which requires
    strict geometry + settled + gripper open AND, history-gated), this is a smooth
    product so PPO gets a dense signal for the final precise descent and hold that
    produces a clean stack. This closes the "releases but never settles" gap seen
    in the M9-A4 strict eval (geometry reached 5/128, released 0/128, settled 0/128).
    """
    pstate = read_placement_state(env, gripper_open_tolerance_m=gripper_open_tolerance_m)
    xy_error = torch.linalg.vector_norm(
        pstate.red_pos_w[..., :2] - pstate.blue_pos_w[..., :2], dim=-1
    )
    z_error = torch.abs(
        pstate.red_pos_w[..., 2] - (pstate.blue_pos_w[..., 2] + float(target_height_diff_m))
    )
    red_speed = torch.linalg.vector_norm(pstate.red_lin_vel_w, dim=-1)
    xy_score = smooth_distance_reward(xy_error, std_m=float(xy_std_m))
    z_score = smooth_distance_reward(z_error, std_m=float(z_std_m))
    vel_score = smooth_distance_reward(red_speed, std_m=float(vel_std_m))
    return (xy_score * z_score * vel_score).to(device=env.device)
