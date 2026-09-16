"""M23 vector environment: PPO alone chooses all PLACE actions.

Snapshots are only an initial-state curriculum, never an online action source.
No old actor, BC checkpoint, phase tracker, release latch, or geometry controller
is imported. Success criteria remain the old XY/Z/open/speed/5-frame contract.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from stage_vla.envs.state_readers import read_placement_state, read_grasp_state, to_torch
from stage_vla.rl.place_snapshot import load_place_snapshot, expand_single_env_state, randomize_rigid_object_xy
from stage_vla.rl.direct_place_core import (
    ACTION_DIM, OBS_DIM, DirectPlaceConfig, direct_action, direct_reward, potential,
    terminal_flags, task_errors, validate_training_snapshot,
)


class DirectPlaceVecEnv(RslRlVecEnvWrapper):
    def __init__(self, env, *, snapshots, cfg=None, xy_jitter_m=.005, seed=17):
        self.task_cfg = cfg or DirectPlaceConfig()
        self.task_cfg.validate()
        if not 0 <= xy_jitter_m <= .03:
            raise ValueError("initial curriculum jitter must be in [0,.03] m")
        if float(env.unwrapped.cfg.actions.arm_action.scale) != 1.:
            raise ValueError("direct metric actions require arm_action.scale=1.0")
        super().__init__(env, clip_actions=1.)
        self.num_actions = ACTION_DIM
        self.max_episode_length = self.task_cfg.episode_steps
        self.xy_jitter_m = float(xy_jitter_m)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.payloads, self.snapshot_paths = [], []
        for path in snapshots:
            payload = load_place_snapshot(path)
            validate_training_snapshot(payload)
            self.payloads.append(payload)
            self.snapshot_paths.append(str(Path(path).resolve()))
        if not self.payloads:
            raise ValueError("no training snapshots")
        self.scene = self.unwrapped.scene
        self.robot = self.scene["robot"]
        self.red = self.scene["cube_2"]
        self.blue = self.scene["cube_1"]
        self.arm_ids, _ = self.robot.find_joints(["panda_joint[1-7]"], preserve_order=True)
        if len(self.arm_ids) != 7:
            raise ValueError("expected 7 arm joints")
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stable_count = torch.zeros_like(self.steps)
        self.bad_count = torch.zeros_like(self.steps)
        self.prev_unit = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.last_source_idx = torch.zeros_like(self.steps)
        self.last_offsets = torch.zeros(self.num_envs, 2, device=self.device)
        self.auto_reset = True
        self.measured = None
        self.total_steps = 0
        self.completed = self.successes = self.failures = self.timeouts = 0
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_fail = self.last_success.clone()
        self.last_timeout = self.last_success.clone()
        self.last_raw_action = torch.zeros(self.num_envs, 7, device=self.device)
        self.unwrapped.single_action_space = gym.spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)
        self.unwrapped.action_space = gym.vector.utils.batch_space(self.unwrapped.single_action_space, self.num_envs)
        self.unwrapped.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-10., 10., (OBS_DIM,), dtype=np.float32)})
        self.unwrapped.observation_space = gym.vector.utils.batch_space(self.unwrapped.single_observation_space, self.num_envs)

    def _measure(self):
        p = read_placement_state(self.unwrapped)
        g = read_grasp_state(self.unwrapped)
        # Clone buffers so a simulator step/reset cannot mutate the previous state.
        return {"red": p.red_pos_w.clone(), "blue": p.blue_pos_w.clone(),
                "vel": p.red_lin_vel_w.clone(), "angular": p.red_ang_vel_w.clone(),
                "open": p.gripper_open.clone(), "grip": p.gripper_joint_pos.clone(),
                "speed": p.red_lin_vel_w.norm(dim=-1), "ee": g.ee_pos_w.clone(),
                "force": torch.stack([g.finger_a_force_n, g.finger_b_force_n], -1),
                "red_quat": to_torch(self.red.data.root_quat_w).clone(),
                "blue_quat": to_torch(self.blue.data.root_quat_w).clone(),
                "q": to_torch(self.robot.data.joint_pos)[:, self.arm_ids].clone(),
                "qd": to_torch(self.robot.data.joint_vel)[:, self.arm_ids].clone()}

    def _restore(self, ids):
        if not len(ids):
            return
        choices = torch.randint(len(self.payloads), (len(ids),), generator=self.rng)
        offsets = (torch.rand(len(ids), 2, generator=self.rng) * 2 - 1) * self.xy_jitter_m
        for choice in choices.unique().tolist():
            mask = choices == choice
            selected = ids[mask.to(self.device)]
            state = expand_single_env_state(self.payloads[choice]["scene_state"], len(selected), self.device)
            delta = offsets[mask].to(self.device)
            randomize_rigid_object_xy(state, "cube_1", delta)
            self.unwrapped.reset_to(state, env_ids=selected, is_relative=True)
            self.last_source_idx[selected] = choice
            self.last_offsets[selected] = delta
        self.steps[ids] = self.stable_count[ids] = self.bad_count[ids] = 0
        self.prev_unit[ids] = 0
        self.returns[ids] = 0
        self.unwrapped.episode_length_buf[ids] = 0

    def _obs(self):
        m, c = self.measured, self.task_cfg
        rel = m["red"] - m["blue"]
        rel = rel.clone()
        rel[:, 2] -= c.stack_height_m
        obs = torch.cat([rel / .05, (m["ee"] - m["red"]) / .05,
                         m["vel"] / .2, m["angular"] / 2., m["red_quat"], m["blue_quat"],
                         m["grip"] / .04, m["force"] / 10., m["q"] / 3., m["qd"] / 2.,
                         self.prev_unit, (self.stable_count / c.stable_steps).unsqueeze(1),
                         (self.bad_count / c.failure_consecutive_steps).unsqueeze(1),
                         (self.steps / c.episode_steps).unsqueeze(1)], dim=-1)
        if obs.shape != (self.num_envs, OBS_DIM) or not torch.isfinite(obs).all():
            raise RuntimeError(f"invalid direct observation {obs.shape}")
        return TensorDict({"policy": obs.clamp(-10, 10)}, batch_size=[self.num_envs])

    def reset(self):
        self._restore(torch.arange(self.num_envs, device=self.device))
        self.measured = self._measure()
        xy, _ = task_errors(self.measured["red"], self.measured["blue"], self.task_cfg)
        height = self.measured["red"][:, 2] - self.measured["blue"][:, 2]
        if not ((height > self.task_cfg.fallen_height_diff_m) & (xy < self.task_cfg.far_xy_m)).all():
            raise RuntimeError("restored training state is not a valid PLACE entry")
        return self._obs(), {}

    def get_observations(self):
        if self.measured is None:
            return self.reset()[0]
        return self._obs()

    def step(self, actions):
        if self.measured is None:
            self.reset()
        c, m = self.task_cfg, self.measured
        before = potential(m["red"], m["blue"], m["speed"], m["open"], c)
        raw, unit = direct_action(torch.as_tensor(actions, device=self.device, dtype=torch.float32), c)
        self.env.step(raw)  # Base reward/termination managers do not decide this task.
        self.last_raw_action = raw.detach().clone()
        self.prev_unit.copy_(unit)
        self.steps += 1
        m = self._measure()
        success, fail, timeout, next_stable, next_bad, strict = terminal_flags(
            m["red"], m["blue"], m["speed"], m["open"], self.stable_count, self.bad_count, self.steps, c)
        # Keep the original normal tensors: RSL collects under inference_mode,
        # but evaluation/training reset boundaries can run outside that context.
        self.stable_count.copy_(next_stable)
        self.bad_count.copy_(next_bad)
        after = potential(m["red"], m["blue"], m["speed"], m["open"], c)
        reward, parts = direct_reward(before, after, unit, success, fail, timeout, c)
        done = success | fail | timeout
        self.last_success, self.last_fail, self.last_timeout = success.clone(), fail.clone(), timeout.clone()
        self.returns += reward
        self.measured = m
        extras = {"log": {"m23/reward": reward.mean(), "m23/open_command": (raw[:, 6] > 0).float().mean(),
                          "m23/actual_open": m["open"].float().mean(), "m23/strict_now": strict.float().mean(),
                          **{f"m23/reward_{key}": value.mean() for key, value in parts.items()}}}
        ids = done.nonzero(as_tuple=False).flatten()
        if self.auto_reset:
            self.total_steps += self.num_envs
            self.completed += len(ids)
            self.successes += int(success.sum().item())
            self.failures += int(fail.sum().item())
            self.timeouts += int(timeout.sum().item())
            if len(ids):
                extras["log"].update({"m23/episode_return": self.returns[ids].mean(),
                                      "m23/episode_success": success[ids].float().mean(),
                                      "m23/episode_length": self.steps[ids].float().mean()})
                self._restore(ids)
                self.measured = self._measure()
        # All endings, including the task's finite time budget, are terminal.
        # Time-left is observed. No RSL timeout bootstrap is requested.
        return self._obs(), reward, done, extras

    def statistics(self):
        return {"transitions": self.total_steps, "completed_episodes": self.completed,
                "successes": self.successes, "physical_failures": self.failures, "timeouts": self.timeouts,
                "training_episode_success_rate": self.successes / self.completed if self.completed else None,
                "contract": asdict(self.task_cfg)}
