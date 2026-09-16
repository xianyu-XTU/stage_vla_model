"""Vector environment for the independent StagePPO DESCEND skill."""
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from stage_vla.envs.state_readers import read_grasp_state
from .direct_place_core import direct_action
from .direct_place_vecenv import DirectPlaceVecEnv
from .place_snapshot import expand_single_env_state, randomize_rigid_object_xy, scale_scene_velocities
from .stageppo_align_v2_core import align_v2_entry_geometry
from .stageppo_align_v3_vecenv import control_step_speed
from .stageppo_descend_core import (
    DescendConfig, descend_errors, descend_flags, descend_interface,
    descend_reward, validate_descend_snapshot,
)


class DescendVecEnv(DirectPlaceVecEnv):
    def __init__(self, env, *, snapshots, cfg=None, xy_jitter_m=0.005, seed=48,
                 initial_velocity_scale=1.0, initial_velocity_scale_min=None,
                 source_weights=None):
        super().__init__(env, snapshots=snapshots, cfg=cfg or DescendConfig(),
                         xy_jitter_m=xy_jitter_m, seed=seed)
        for payload in self.payloads:
            validate_descend_snapshot(payload)
        if initial_velocity_scale_min is None:
            initial_velocity_scale_min = initial_velocity_scale
        if not (0.0 <= float(initial_velocity_scale_min) <= float(initial_velocity_scale) <= 1.0):
            raise ValueError("DESCEND initial velocity scale range must lie in [0,1]")
        self.initial_velocity_scale_min = float(initial_velocity_scale_min)
        self.initial_velocity_scale = float(initial_velocity_scale)
        if {payload["source_seed"] for payload in self.payloads} != {1004, 1009, 1015, 1020, 1031}:
            raise ValueError("DESCEND requires all five unique TRAIN sources")
        self.weighted_source_sampling = source_weights is not None
        if source_weights is None:
            source_weights = [1.0] * len(self.payloads)
        self.source_weights = torch.as_tensor(source_weights, dtype=torch.float32)
        if (self.source_weights.shape != (len(self.payloads),)
                or not torch.isfinite(self.source_weights).all()
                or not (self.source_weights > 0).all()):
            raise ValueError("DESCEND source weights must contain one positive finite value per snapshot")
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_velocity_scale = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.restore_calls = 0
        self.last_diagnostics = None
        self.control_step_dt = float(self.unwrapped.step_dt)
        self.interface = descend_interface(self.task_cfg, self.control_step_dt)
        self.unwrapped.single_observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Box(-10.0, 10.0, (52,), dtype=np.float32)
        })
        self.unwrapped.observation_space = gym.vector.utils.batch_space(
            self.unwrapped.single_observation_space, self.num_envs
        )

    def _measure(self):
        measured = super()._measure()
        grasp = read_grasp_state(self.unwrapped)
        measured.update(left_tip=grasp.left_tip_w.clone(), right_tip=grasp.right_tip_w.clone())
        return measured

    def _restore(self, ids):
        if not len(ids):
            return
        self.restore_calls += 1
        count = len(self.payloads)
        if self.weighted_source_sampling:
            choices = torch.multinomial(
                self.source_weights, len(ids), replacement=True, generator=self.rng
            )
        else:
            choices = ((torch.arange(len(ids)) % count)[torch.randperm(len(ids), generator=self.rng)]
                       if len(ids) >= count else torch.randint(count, (len(ids),), generator=self.rng))
        offsets = (torch.rand(len(ids), 2, generator=self.rng) * 2 - 1) * self.xy_jitter_m
        if self.initial_velocity_scale_min == self.initial_velocity_scale:
            velocity_scales = torch.full((len(ids),), self.initial_velocity_scale)
        else:
            velocity_scales = self.initial_velocity_scale_min + (
                self.initial_velocity_scale - self.initial_velocity_scale_min
            ) * torch.rand(len(ids), generator=self.rng)
        for choice in choices.unique().tolist():
            mask = choices == choice
            selected = ids[mask.to(self.device)]
            state = expand_single_env_state(self.payloads[choice]["scene_state"], len(selected), self.device)
            scale_scene_velocities(state, velocity_scales[mask])
            delta = offsets[mask].to(self.device)
            randomize_rigid_object_xy(state, "cube_1", delta)
            self.unwrapped.reset_to(state, env_ids=selected, is_relative=True)
            self.last_source_idx[selected] = choice
            self.last_offsets[selected] = delta
            self.last_velocity_scale[selected] = velocity_scales[mask].to(self.device)
        self.steps[ids] = self.stable_count[ids] = self.bad_count[ids] = 0
        self.prev_unit[ids] = 0
        self.returns[ids] = 0
        self.finished[ids] = False
        self.unwrapped.episode_length_buf[ids] = 0
        measured = self._measure()
        geometry = align_v2_entry_geometry(measured)
        xy, height, _ = descend_errors(measured, self.task_cfg)
        valid = (geometry & (height > self.task_cfg.stack_height_m + self.task_cfg.z_success_m)
                 & (height < 0.10) & (xy < self.task_cfg.far_xy_m))
        if not valid[ids].all():
            raise RuntimeError(f"invalid DESCEND restored entries: {ids[~valid[ids]].tolist()}")

    def _obs(self):
        measured, cfg = self.measured, self.task_cfg
        relative = (measured["red"] - measured["blue"]).clone()
        relative[:, 2] -= cfg.stack_height_m
        obs = torch.cat([
            relative / 0.05, (measured["ee"] - measured["red"]) / 0.05,
            measured["vel"] / 0.2, measured["angular"] / 2.0,
            measured["red_quat"], measured["blue_quat"], measured["grip"] / 0.04,
            measured["force"] / 10.0, measured["q"] / 3.0, measured["qd"] / 2.0,
            self.prev_unit, (self.stable_count / cfg.stable_steps).unsqueeze(1),
            (self.bad_count / cfg.failure_consecutive_steps).unsqueeze(1),
            (self.steps / cfg.episode_steps).unsqueeze(1),
            (measured["left_tip"] - measured["red"]) / 0.05,
            (measured["right_tip"] - measured["red"]) / 0.05,
        ], dim=-1)
        if obs.shape != (self.num_envs, 52) or not torch.isfinite(obs).all():
            raise RuntimeError("invalid 52D DESCEND observation")
        return TensorDict({"policy": obs.clamp(-10, 10)}, batch_size=[self.num_envs])

    def reset(self):
        self._restore(torch.arange(self.num_envs, device=self.device))
        self.measured = self._measure()
        return self._obs(), {}

    def step(self, actions):
        if self.measured is None:
            self.reset()
        active = ~self.finished
        before, cfg = self.measured, self.task_cfg
        raw, unit = direct_action(torch.as_tensor(actions, device=self.device, dtype=torch.float32), cfg)
        self.env.step(raw)
        if self.unwrapped.reset_buf.any():
            raise RuntimeError("base simulator reset during DESCEND step")
        self.last_raw_action.copy_(raw)
        self.prev_unit[active] = unit[active]
        self.steps += active.long()
        after = self._measure()
        after["instantaneous_speed"] = after["speed"]
        after["speed"] = control_step_speed(before["red"], after["red"], self.control_step_dt)
        success, failure, timeout, stable, bad, diagnostics = descend_flags(
            after, self.stable_count, self.bad_count, self.steps, cfg
        )
        success, failure, timeout = success & active, failure & active, timeout & active
        self.stable_count[active] = stable[active]
        self.bad_count[active] = bad[active]
        self.last_success.copy_(success)
        self.last_fail.copy_(failure)
        self.last_timeout.copy_(timeout)
        self.last_diagnostics = diagnostics
        done = success | failure | timeout
        reward = descend_reward(before, after, unit, success, failure, timeout, cfg) * active
        self.returns += reward
        self.finished |= done
        self.measured = after
        extras = {"log": {
            "descend/reward": reward.mean(), "descend/held": diagnostics["held"].float().mean(),
            "descend/strict": diagnostics["strict"].float().mean(),
            "descend/xy_m": diagnostics["xy"].mean(),
            "descend/z_error_m": diagnostics["z_error"].mean(),
            "descend/control_speed_mps": after["speed"].mean(),
        }}
        if self.auto_reset:
            ids = done.nonzero(as_tuple=False).flatten()
            self.total_steps += self.num_envs
            self.completed += len(ids)
            self.successes += int(success.sum())
            self.failures += int(failure.sum())
            self.timeouts += int(timeout.sum())
            if len(ids):
                self._restore(ids)
                self.measured = self._measure()
        return self._obs(), reward, done, extras
