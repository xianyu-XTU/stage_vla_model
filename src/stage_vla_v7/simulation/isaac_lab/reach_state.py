"""Isaac state sampling for the frozen cube REACH policy."""

from __future__ import annotations

from .state_reader import read_grasp_state, read_placement_state, to_torch


def measure_reach_state(
    base_env,
    *,
    object_asset_name: str = "cube_2",
    support_asset_name: str = "cube_1",
):
    """Measure a color-agnostic object/support pair from an Isaac scene."""
    placement = read_placement_state(
        base_env,
        object_asset_name=object_asset_name,
        support_asset_name=support_asset_name,
    )
    grasp = read_grasp_state(base_env, object_asset_name=object_asset_name)
    robot = base_env.scene["robot"]
    arm_ids, _ = robot.find_joints(
        ["panda_joint[1-7]"], preserve_order=True
    )
    return {
        "ee": grasp.end_effector_position_w.clone(),
        "red": placement.object_position_w.clone(),
        "blue": placement.support_position_w.clone(),
        "left_tip": grasp.left_fingertip_position_w.clone(),
        "right_tip": grasp.right_fingertip_position_w.clone(),
        "speed": placement.object_linear_velocity_w.norm(dim=-1),
        "red_vel": placement.object_linear_velocity_w.clone(),
        "red_ang": placement.object_angular_velocity_w.clone(),
        "open": placement.gripper_open.clone(),
        "grip": placement.gripper_joint_position.clone(),
        "q": to_torch(robot.data.joint_pos)[:, arm_ids].clone(),
        "qd": to_torch(robot.data.joint_vel)[:, arm_ids].clone(),
        "red_quat": to_torch(
            base_env.scene[object_asset_name].data.root_quat_w
        ).clone(),
        "blue_quat": to_torch(
            base_env.scene[support_asset_name].data.root_quat_w
        ).clone(),
    }
