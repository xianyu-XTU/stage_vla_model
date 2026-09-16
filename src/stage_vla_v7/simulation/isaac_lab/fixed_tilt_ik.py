"""Isaac IK action that holds reset-time roll/pitch and preserves policy yaw."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)

from ..physics.fixed_tilt import FixedTiltReference


class FixedTiltDifferentialInverseKinematicsAction(
    DifferentialInverseKinematicsAction
):
    """Relative XYZ/yaw IK with a per-environment reset-time roll/pitch target."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if (
            self.action_dim != 6
            or not cfg.controller.use_relative_mode
            or cfg.controller.command_type != "pose"
        ):
            raise ValueError(
                "fixed-tilt action requires the six-dimensional relative pose controller"
            )
        self.tilt_reference = FixedTiltReference(self.num_envs, self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        if not torch.isfinite(self._processed_actions).all():
            raise ValueError("IK action contains non-finite values")
        if (self._processed_actions[:, 3:5].abs() > 1e-8).any():
            raise ValueError("policy roll/pitch commands are outside the fixed-tilt contract")
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        target_quat = self.tilt_reference.target(
            ee_quat_curr, self._processed_actions[:, 5]
        )
        self._ik_controller.set_command(
            self._processed_actions, ee_pos_curr, ee_quat_curr
        )
        self._ik_controller.ee_quat_des[:] = target_quat

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        self.tilt_reference.reset(env_ids)

    @property
    def desired_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des
