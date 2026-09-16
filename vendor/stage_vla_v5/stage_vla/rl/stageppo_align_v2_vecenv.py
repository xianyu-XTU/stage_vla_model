"""ALIGN-v2 vector environment for fixed-tilt demonstration/RL experiments."""
import torch

from .direct_place_core import direct_action
from .place_snapshot import expand_single_env_state, randomize_rigid_object_xy
from .stageppo_align_core import AlignConfig, align_errors
from .stageppo_align_vecenv import AlignVecEnv
from .stageppo_align_v2_core import (
    align_v2_entry_geometry, align_v2_flags, align_v2_interface, align_v2_reward,
)


class AlignV2VecEnv(AlignVecEnv):
    def __init__(self, env, *, snapshots, cfg=None, xy_jitter_m=.025, seed=31):
        super().__init__(env, snapshots=snapshots, cfg=cfg or AlignConfig(),
                         xy_jitter_m=xy_jitter_m, seed=seed)
        self.interface = align_v2_interface(self.task_cfg, float(self.unwrapped.step_dt))

    def _restore(self, ids):
        if not len(ids):
            return
        self.restore_calls += 1
        choices = ((torch.arange(len(ids)) % 5)[torch.randperm(len(ids), generator=self.rng)]
                   if len(ids) >= 5 else torch.randint(5, (len(ids),), generator=self.rng))
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
        measured = self._measure()
        geometry = align_v2_entry_geometry(measured)
        xy, height, _ = align_errors(measured, self.task_cfg)
        valid = (geometry & (height > self.task_cfg.height_failure_m) & (height < .15)
                 & (xy < self.task_cfg.far_xy_m))
        if not valid[ids].all():
            failed = ids[~valid[ids]].tolist()
            raise RuntimeError(f"invalid ALIGN-v2 restored entries: {failed}")

    def step(self, actions):
        if self.measured is None:
            self.reset()
        active = ~self.finished
        before, cfg = self.measured, self.task_cfg
        raw, unit = direct_action(torch.as_tensor(actions, device=self.device, dtype=torch.float32), cfg)
        self.env.step(raw)
        if self.unwrapped.reset_buf.any():
            raise RuntimeError("base simulator reset during ALIGN-v2 step")
        self.last_raw_action.copy_(raw)
        self.prev_unit[active] = unit[active]
        self.steps += active.long()
        after = self._measure()
        success, failure, timeout, stable, bad, diagnostics = align_v2_flags(
            after, self.stable_count, self.bad_count, self.steps, cfg)
        success, failure, timeout = success & active, failure & active, timeout & active
        self.stable_count[active] = stable[active]
        self.bad_count[active] = bad[active]
        self.last_success.copy_(success)
        self.last_fail.copy_(failure)
        self.last_timeout.copy_(timeout)
        self.last_diagnostics = diagnostics
        done = success | failure | timeout
        reward = align_v2_reward(before, after, unit, success, failure, timeout, cfg) * active
        self.returns += reward
        self.finished |= done
        self.measured = after
        extras = {"log": {"align_v2/reward": reward.mean(),
            "align_v2/held": diagnostics["held"].float().mean(),
            "align_v2/strict": diagnostics["strict"].float().mean(),
            "align_v2/xy_m": diagnostics["xy"].mean()}}
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
