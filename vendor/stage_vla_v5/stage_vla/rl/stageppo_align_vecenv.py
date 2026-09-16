"""Independent ALIGN PPO: policy owns XYZ/yaw/grip, no action prefix or latch."""
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from stage_vla.envs.state_readers import read_grasp_state
from .direct_place_core import direct_action
from .direct_place_vecenv import DirectPlaceVecEnv
from .place_snapshot import expand_single_env_state, randomize_rigid_object_xy
from .stageppo_align_core import (
    AlignConfig, align_errors, align_flags, align_grasp, align_interface,
    align_reward, validate_align_snapshot,
)


class AlignVecEnv(DirectPlaceVecEnv):
    def __init__(self, env, *, snapshots, cfg=None, xy_jitter_m=.025, seed=28):
        super().__init__(env, snapshots=snapshots, cfg=cfg or AlignConfig(), xy_jitter_m=xy_jitter_m, seed=seed)
        for p in self.payloads:
            validate_align_snapshot(p)
        if {p["source_seed"] for p in self.payloads} != {1004, 1009, 1015, 1020, 1031} or len(self.payloads) != 5:
            raise ValueError("all five unique TRAIN sources required")
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.restore_calls = 0
        self.last_diagnostics = None
        self.interface = align_interface(self.task_cfg, float(self.unwrapped.step_dt))
        self.unwrapped.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-10., 10., (52,), dtype=np.float32)})
        self.unwrapped.observation_space = gym.vector.utils.batch_space(self.unwrapped.single_observation_space, self.num_envs)

    def _measure(self):
        m = super()._measure()
        g = read_grasp_state(self.unwrapped)
        m.update(left_tip=g.left_tip_w.clone(), right_tip=g.right_tip_w.clone())
        return m

    def _restore(self, ids):
        if not len(ids):
            return
        self.restore_calls += 1
        # Every full evaluation batch covers all five sources; no failed source
        # can silently disappear. Small asynchronous reset batches are random.
        choices = (torch.arange(len(ids)) % 5)[torch.randperm(len(ids), generator=self.rng)] if len(ids) >= 5 else torch.randint(5, (len(ids),), generator=self.rng)
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
        self.finished[ids] = False
        self.unwrapped.episode_length_buf[ids] = 0
        m = self._measure()
        _, geometry, _ = align_grasp(m)
        xy, height, _ = align_errors(m, self.task_cfg)
        valid = geometry & (height > self.task_cfg.height_failure_m) & (height < .15) & (xy < self.task_cfg.far_xy_m)
        if not valid[ids].all():
            raise RuntimeError("invalid restored ALIGN entry; no hidden warmup or source replacement permitted")

    def _obs(self):
        m, c = self.measured, self.task_cfg
        rel = (m["red"] - m["blue"]).clone()
        rel[:, 2] -= c.height_target_m
        obs = torch.cat([rel / .05, (m["ee"] - m["red"]) / .05,
            m["vel"] / .2, m["angular"] / 2., m["red_quat"], m["blue_quat"],
            m["grip"] / .04, m["force"] / 10., m["q"] / 3., m["qd"] / 2.,
            self.prev_unit, (self.stable_count / c.stable_steps).unsqueeze(1),
            (self.bad_count / c.failure_consecutive_steps).unsqueeze(1),
            (self.steps / c.episode_steps).unsqueeze(1),
            (m["left_tip"] - m["red"]) / .05, (m["right_tip"] - m["red"]) / .05], -1)
        if obs.shape != (self.num_envs, 52) or not torch.isfinite(obs).all():
            raise RuntimeError("invalid 52D ALIGN observation")
        return TensorDict({"policy": obs.clamp(-10, 10)}, batch_size=[self.num_envs])

    def reset(self):
        self._restore(torch.arange(self.num_envs, device=self.device))
        self.measured = self._measure()
        return self._obs(), {}

    def step(self, actions):
        if self.measured is None:
            self.reset()
        active = ~self.finished
        before, c = self.measured, self.task_cfg
        raw, unit = direct_action(torch.as_tensor(actions, device=self.device, dtype=torch.float32), c)
        self.env.step(raw)
        if self.unwrapped.reset_buf.any():
            raise RuntimeError("base simulator reset during ALIGN step")
        self.last_raw_action.copy_(raw)
        self.prev_unit[active] = unit[active]
        self.steps += active.long()
        after = self._measure()
        s, f, t, stable, bad, d = align_flags(after, self.stable_count, self.bad_count, self.steps, c)
        s, f, t = s & active, f & active, t & active
        self.stable_count[active] = stable[active]
        self.bad_count[active] = bad[active]
        self.last_success.copy_(s)
        self.last_fail.copy_(f)
        self.last_timeout.copy_(t)
        self.last_diagnostics = d
        done = s | f | t
        reward = align_reward(before, after, unit, s, f, t, c) * active
        self.returns += reward
        self.finished |= done
        self.measured = after
        extras = {"log": {"align/reward": reward.mean(), "align/held": d["held"].float().mean(),
            "align/strict": d["strict"].float().mean(), "align/xy_m": d["xy"].mean(),
            "align/open_command": (raw[:, 6] > 0).float().mean()}}
        if self.auto_reset:
            ids = done.nonzero(as_tuple=False).flatten()
            self.total_steps += self.num_envs
            self.completed += len(ids)
            self.successes += int(s.sum())
            self.failures += int(f.sum())
            self.timeouts += int(t.sum())
            if len(ids):
                extras["log"].update({"align/episode_return": self.returns[ids].mean(),
                    "align/episode_success": s[ids].float().mean(), "align/episode_length": self.steps[ids].float().mean()})
                self._restore(ids)
                self.measured = self._measure()
        # Finite terminal horizon, no time_outs bootstrap and no reward after
        # the first terminal event in an evaluation row.
        return self._obs(), reward, done, extras
