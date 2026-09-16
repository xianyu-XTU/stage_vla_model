"""ALIGN-v3 uses control-step displacement for terminal linear speed."""
from __future__ import annotations

import torch

from .direct_place_core import direct_action
from .stageppo_align_v2_core import align_v2_flags, align_v2_reward
from .stageppo_align_v2_vecenv import AlignV2VecEnv


def control_step_speed(before_position, after_position, step_dt):
    if before_position.shape != after_position.shape or before_position.ndim != 2:
        raise ValueError("positions must have matching [N,D] shapes")
    if step_dt <= 0:
        raise ValueError("control step duration must be positive")
    return (after_position - before_position).norm(dim=-1) / step_dt


class AlignV3VecEnv(AlignV2VecEnv):
    """Preserve ALIGN-v2 geometry while rejecting instantaneous contact jitter."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.control_step_dt = float(self.unwrapped.step_dt)
        self.interface = {
            **self.interface,
            "version": "stageppo-align-v3-control-speed-state52-action5-v1",
            "success_linear_speed": "norm(red_position[t]-red_position[t-1])/control_step_dt",
            "instantaneous_physx_velocity_used_for_success": False,
        }

    def step(self, actions):
        if self.measured is None:
            self.reset()
        active = ~self.finished
        before, cfg = self.measured, self.task_cfg
        raw, unit = direct_action(torch.as_tensor(actions, device=self.device, dtype=torch.float32), cfg)
        self.env.step(raw)
        if self.unwrapped.reset_buf.any():
            raise RuntimeError("base simulator reset during ALIGN-v3 step")
        self.last_raw_action.copy_(raw)
        self.prev_unit[active] = unit[active]
        self.steps += active.long()
        after = self._measure()
        after["instantaneous_speed"] = after["speed"]
        after["speed"] = control_step_speed(before["red"], after["red"], self.control_step_dt)
        success, failure, timeout, stable, bad, diagnostics = align_v2_flags(
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
        reward = align_v2_reward(before, after, unit, success, failure, timeout, cfg) * active
        self.returns += reward
        self.finished |= done
        self.measured = after
        extras = {"log": {
            "align_v3/reward": reward.mean(), "align_v3/held": diagnostics["held"].float().mean(),
            "align_v3/strict": diagnostics["strict"].float().mean(),
            "align_v3/xy_m": diagnostics["xy"].mean(),
            "align_v3/control_speed_mps": after["speed"].mean(),
            "align_v3/instantaneous_speed_mps": after["instantaneous_speed"].mean(),
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
