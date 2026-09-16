"""RSL-RL wrapper for the M9-B factorized Action DSL.

The PPO actor outputs four integer-valued categorical factors.  This wrapper
performs the deterministic DSL decode, reuses the exact M9-A edge-aligned yaw
controller, and finally delegates the 7-D raw pose-relative IK action to the
standard Isaac Lab RSL-RL wrapper.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .action_adapter import M9A_RAW_ACTION_DIM
from .action_dsl import M9B_CATEGORY_COUNTS, M9B_POLICY_FACTORS, M9BDecodedAction, decode_m9b_tokens
from .edge_alignment import EdgeAlignmentConfig, edge_aligned_raw_action, load_edge_alignment_cfg


class M9BActionDSLVecEnvWrapper(RslRlVecEnvWrapper):
    """Expose four categorical DSL factors over the frozen A4 low-level control stack."""

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        *,
        config_file: Path | None = None,
        edge_alignment_cfg: EdgeAlignmentConfig | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        raw_dim = int(self.unwrapped.action_manager.total_action_dim)
        if raw_dim != M9A_RAW_ACTION_DIM:
            raise RuntimeError(
                f"M9-B DSL adapter expects raw ActionManager dim {M9A_RAW_ACTION_DIM}, got {raw_dim}."
            )
        if edge_alignment_cfg is None:
            if config_file is None:
                raise ValueError("config_file or edge_alignment_cfg is required for deterministic yaw alignment")
            edge_alignment_cfg = load_edge_alignment_cfg(config_file)
        self.edge_alignment_cfg = edge_alignment_cfg

        self.num_actions = M9B_POLICY_FACTORS
        self.category_counts = M9B_CATEGORY_COUNTS
        self.last_dsl_decode: M9BDecodedAction | None = None
        self.last_edge_alignment = None

        # Every freshly reset Franka starts from the open-gripper command state.
        self._gripper_raw_memory = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)

        # Metadata describes the logical discrete policy interface; the underlying
        # Isaac ActionManager intentionally remains the same 7-D Box as M9-A4.
        self.unwrapped.single_action_space = gym.spaces.MultiDiscrete(np.asarray(self.category_counts, dtype=np.int64))
        self.unwrapped.action_space = gym.vector.utils.batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )

    @property
    def gripper_raw_memory(self):
        return self._gripper_raw_memory.clone()

    def _before_env_step(self, decoded: M9BDecodedAction) -> None:
        """Subclass hook after DSL decode and before the raw Isaac env step.

        M9-B leaves this as a no-op. M10 uses it to expose the selected GRIP
        token to the stateful reward term.
        """
        return None

    def _transform_raw_actions(self, decoded: M9BDecodedAction, raw_actions: torch.Tensor) -> torch.Tensor:
        """Optional subclass hook for a final raw-action transform.

        The base M9-B implementation is an exact no-op.  M10-11-H4X overrides
        only this hook to suppress terminal XYZ translation while preserving
        the shared DSL decoder, deterministic edge yaw and GRIP semantics.
        """
        return raw_actions

    def step(self, actions):
        action_tensor = torch.as_tensor(actions, device=self.device)
        decoded = decode_m9b_tokens(action_tensor, self._gripper_raw_memory)
        raw_actions, runtime = edge_aligned_raw_action(
            self.unwrapped,
            decoded.logical_action,
            self.edge_alignment_cfg,
        )
        raw_actions = self._transform_raw_actions(decoded, raw_actions)
        self._before_env_step(decoded)
        self.last_dsl_decode = decoded
        self.last_edge_alignment = runtime
        out = super().step(raw_actions)

        dones = out[2]
        done_mask = torch.as_tensor(dones, device=self.device, dtype=torch.bool).reshape(-1)
        next_memory = decoded.next_gripper_raw.to(device=self.device, dtype=torch.float32)
        self._gripper_raw_memory = torch.where(done_mask, torch.ones_like(next_memory), next_memory)
        return out
