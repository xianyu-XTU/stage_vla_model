"""RSL-RL wrapper exposing 4 learned actions over 7-D edge-aligned pose IK.

The policy sees only ``[dx,dy,dz,gripper]``.  Immediately before every base
step, a deterministic controller computes a bounded yaw correction that aligns
the real fingertip closing axis to the nearest red-cube local X/Y axis.  The
resulting raw Isaac action is::

    [dx, dy, dz, 0, 0, deterministic_dRz, gripper]

The same deterministic yaw controller is intended to be shared by M9-A and
M9-B so the later comparison changes only the learned action representation.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .action_adapter import M9A_POLICY_ACTION_DIM, M9A_RAW_ACTION_DIM
from .edge_alignment import EdgeAlignmentConfig, edge_aligned_raw_action, load_edge_alignment_cfg


class M9AFourDVecEnvWrapper(RslRlVecEnvWrapper):
    """Expose 4-D PPO actions while injecting deterministic edge-aligned yaw."""

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        *,
        config_file: Path | None = None,
        edge_alignment_cfg: EdgeAlignmentConfig | None = None,
    ):
        super().__init__(env, clip_actions=clip_actions)
        raw_dim = int(self.unwrapped.action_manager.total_action_dim)
        if raw_dim != M9A_RAW_ACTION_DIM:
            raise RuntimeError(
                f"M9-A pose adapter expects raw ActionManager dim {M9A_RAW_ACTION_DIM}, got {raw_dim}."
            )
        if edge_alignment_cfg is None:
            if config_file is None:
                raise ValueError("config_file or edge_alignment_cfg is required for deterministic yaw alignment")
            edge_alignment_cfg = load_edge_alignment_cfg(config_file)
        self.edge_alignment_cfg = edge_alignment_cfg
        self.last_edge_alignment = None

        # RSL-RL reads this attribute when building the actor output head.
        self.num_actions = M9A_POLICY_ACTION_DIM

        # Keep public action-space metadata consistent with the logical policy
        # interface.  The underlying ActionManager itself remains 7-D.
        low = -float(clip_actions) if clip_actions is not None else -float("inf")
        high = float(clip_actions) if clip_actions is not None else float("inf")
        self.unwrapped.single_action_space = gym.spaces.Box(
            low=low, high=high, shape=(M9A_POLICY_ACTION_DIM,), dtype=np.float32
        )
        self.unwrapped.action_space = gym.vector.utils.batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )

    def step(self, actions):
        raw_actions, runtime = edge_aligned_raw_action(self.unwrapped, actions, self.edge_alignment_cfg)
        self.last_edge_alignment = runtime
        # Parent wrapper performs configured raw-action clipping and then calls
        # the base environment.  dRx/dRy stay exactly zero; dRz is deterministic.
        return super().step(raw_actions)
