"""Lightweight oracle-state REACH observation and action helpers."""

from __future__ import annotations

import torch

from stage_vla.envs.state_readers import read_grasp_state, read_placement_state, to_torch


REACH_OBS_DIM = 52


def measure_reach_state(
    base_env, *, object_asset_name: str = "cube_2",
    support_asset_name: str = "cube_1",
):
    """Measure a color-agnostic object/support pair from an Isaac scene."""
    placement = read_placement_state(
        base_env, red_cube_name=object_asset_name,
        blue_cube_name=support_asset_name,
    )
    grasp = read_grasp_state(base_env, red_cube_name=object_asset_name)
    robot = base_env.scene["robot"]
    arm_ids, _ = robot.find_joints(["panda_joint[1-7]"], preserve_order=True)
    return {
        "ee": grasp.ee_pos_w.clone(),
        "red": placement.red_pos_w.clone(),
        "blue": placement.blue_pos_w.clone(),
        "left_tip": grasp.left_tip_w.clone(),
        "right_tip": grasp.right_tip_w.clone(),
        "speed": placement.red_lin_vel_w.norm(dim=-1),
        "red_vel": placement.red_lin_vel_w.clone(),
        "red_ang": placement.red_ang_vel_w.clone(),
        "open": placement.gripper_open.clone(),
        "grip": placement.gripper_joint_pos.clone(),
        "q": to_torch(robot.data.joint_pos)[:, arm_ids].clone(),
        "qd": to_torch(robot.data.joint_vel)[:, arm_ids].clone(),
        "red_quat": to_torch(base_env.scene[object_asset_name].data.root_quat_w).clone(),
        "blue_quat": to_torch(base_env.scene[support_asset_name].data.root_quat_w).clone(),
    }


def reach_observation(state, previous_action: torch.Tensor) -> torch.Tensor:
    """Encode the 52-D v5 REACH oracle-state contract."""
    n = state["ee"].shape[0]
    target = torch.as_tensor(
        state.get("grasp_target", state["red"]),
        device=state["ee"].device,
        dtype=state["ee"].dtype,
    )
    if target.shape != state["red"].shape:
        raise ValueError("grasp_target must match object position shape")
    previous_action = torch.as_tensor(previous_action, device=state["ee"].device,
                                      dtype=state["ee"].dtype)
    if previous_action.shape != (n, 5):
        raise ValueError("previous_action must have shape [N,5]")
    obs = torch.cat([
        (state["red"] - state["blue"]) / 0.5,
        (state["ee"] - target) / 0.5,
        (state["left_tip"] - target) / 0.1,
        (state["right_tip"] - target) / 0.1,
        state["red_quat"], state["blue_quat"],
        state["q"] / 3.0, state["qd"] / 2.0,
        state["grip"] / 0.04,
        previous_action,
        state["speed"].unsqueeze(-1) / 0.2,
        state["ee"] / 1.0,
        state["red_vel"] / 0.2,
        state["red_ang"] / 2.0,
        state["open"].float().unsqueeze(-1),
    ], dim=-1)
    if obs.shape != (n, REACH_OBS_DIM) or not torch.isfinite(obs).all():
        raise RuntimeError("invalid REACH observation")
    return obs.clamp(-10.0, 10.0)


def reach_raw_action(policy_action: torch.Tensor) -> torch.Tensor:
    """Decode normalized [dx,dy,dz,dyaw,grip] to Isaac's seven actions."""
    action = torch.as_tensor(policy_action, dtype=torch.float32)
    if action.ndim != 2 or action.shape[-1] != 5:
        raise ValueError("REACH policy action must have shape [N,5]")
    unit = action.clamp(-1.0, 1.0)
    raw = torch.zeros((unit.shape[0], 7), device=unit.device, dtype=unit.dtype)
    raw[:, :3] = unit[:, :3] * 0.005
    raw[:, 5] = unit[:, 3] * 0.02
    raw[:, 6] = 1.0
    return raw
