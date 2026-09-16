"""Runtime diagnostics for the TRANSPORT->PLACE live handoff."""
from __future__ import annotations

import torch

from stage_vla.envs.state_readers import read_grasp_state, read_placement_state
from stage_vla.stages import PhysicalGraspConfig, physical_grasp_diagnostics

STRICT_STACK_HEIGHT_DIFF_M = 0.0468


def read_handoff_metrics(base_env) -> dict:
    """Measure geometry + dynamics at a 1-env live handoff/snapshot state."""
    if int(base_env.num_envs) != 1:
        raise ValueError("handoff metrics currently require num_envs=1")
    scene = base_env.scene
    p = read_placement_state(base_env, gripper_open_tolerance_m=0.002)
    g = read_grasp_state(base_env)
    grasp = physical_grasp_diagnostics(
        g.red_pos_w,
        g.left_tip_w,
        g.right_tip_w,
        g.finger_a_force_n,
        g.finger_b_force_n,
        cfg=PhysicalGraspConfig(
            radial_tolerance_m=0.03,
            height_tolerance_m=0.012,
            contact_force_threshold_n=0.5,
            endpoint_margin=0.05,
        ),
    ).physical_grasp
    origin = scene.env_origins[0]
    red = p.red_pos_w[0]
    blue = p.blue_pos_w[0]
    ee = g.ee_pos_w[0]
    target = blue + torch.tensor([0.0, 0.0, 0.04], device=blue.device, dtype=blue.dtype)
    robot = scene["robot"]
    joint_vel = torch.as_tensor(robot.data.joint_vel)[0]

    ee_speed = None
    try:
        body_ids, _ = robot.find_bodies("panda_hand")
        if len(body_ids) == 1:
            body_lin = torch.as_tensor(robot.data.body_lin_vel_w)[0, int(body_ids[0]), :3]
            ee_speed = float(torch.linalg.vector_norm(body_lin).item())
    except Exception:
        ee_speed = None

    gripper_gap = float(torch.as_tensor(p.gripper_joint_pos[0]).sum().item())
    red_speed = float(torch.linalg.vector_norm(p.red_lin_vel_w[0]).item())
    delta_xy = red[:2] - blue[:2]
    xy = float(torch.linalg.vector_norm(delta_xy).item())
    strict_z_err = float((red[2] - (blue[2] + STRICT_STACK_HEIGHT_DIFF_M)).item())
    red_local_z = float((red[2] - origin[2]).item())
    physical = bool(grasp[0].item())
    ready = physical and red_local_z > 0.05 and xy < 0.15

    return {
        "ready": bool(ready),
        "physical_grasp": physical,
        "red_local_z_m": red_local_z,
        "red_blue_xy_m": xy,
        "red_blue_delta_xy_m": delta_xy.detach().cpu().tolist(),
        "strict_z_error_m": strict_z_err,
        "red_speed_mps": red_speed,
        "ee_speed_mps": ee_speed,
        "ee_rel_red_m": (ee - red).detach().cpu().tolist(),
        "ee_rel_target_m": (ee - target).detach().cpu().tolist(),
        "gripper_gap_m": gripper_gap,
        "joint_vel_norm": float(torch.linalg.vector_norm(joint_vel).item()),
        "red_lin_vel_w": p.red_lin_vel_w[0].detach().cpu().tolist(),
        "red_ang_vel_w": p.red_ang_vel_w[0].detach().cpu().tolist(),
    }
