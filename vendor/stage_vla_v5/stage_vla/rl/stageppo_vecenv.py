"""Independent skill PPO training and physically continuous two-skill execution."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .direct_place_core import direct_action, direct_reward, validate_training_snapshot
from .direct_place_vecenv import DirectPlaceVecEnv
from .place_snapshot import load_place_snapshot, expand_single_env_state, randomize_rigid_object_xy
from .stageppo_core import (
    Skill, SkillConfig, chain_transition, entry_valid, interface_contract, skill_flags,
    skill_potential, validate_retreat_snapshot,
)


class StagePPOVecEnv(RslRlVecEnvWrapper):
    # Reuse read-only measurement implementation; old M23 stays unmodified.
    _measure = DirectPlaceVecEnv._measure

    def __init__(self, env, *, snapshots, skill=Skill.RELEASE_STABILIZE, chain=False,
                 cfg=None, xy_jitter_m=.025, seed=17):
        self.task_cfg = cfg or SkillConfig()
        self.task_cfg.validate()
        self.initial_skill = Skill(skill)
        self.chain = bool(chain)
        if chain and self.initial_skill != Skill.RELEASE_STABILIZE:
            raise ValueError("chain must begin with release")
        if not 0 <= xy_jitter_m <= .03 or (self.initial_skill == Skill.RETREAT and xy_jitter_m != 0):
            raise ValueError("retreat entries may not move the supporting cube independently")
        if env.unwrapped.termination_manager.active_terms:
            raise ValueError("disable all base terminations; no hidden auto-reset permitted")
        action_cfg = env.unwrapped.cfg.actions.arm_action
        if float(action_cfg.scale) != 1. or not action_cfg.controller.use_relative_mode:
            raise ValueError("metric relative IK interface required")
        super().__init__(env, clip_actions=1.)
        self.interface = interface_contract(self.task_cfg, float(self.unwrapped.step_dt))
        self.num_actions = 5
        self.max_episode_length = self.task_cfg.episode_steps * (2 if chain else 1)
        self.xy_jitter_m = float(xy_jitter_m)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.snapshot_paths = [str(Path(p).resolve()) for p in snapshots]
        self.payloads = []
        for path in self.snapshot_paths:
            if self.initial_skill == Skill.RELEASE_STABILIZE:
                payload = load_place_snapshot(path)
                validate_training_snapshot(payload)
            else:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                validate_retreat_snapshot(payload, self.task_cfg, self.interface)
            self.payloads.append(payload)
        if not self.payloads:
            raise ValueError("no skill entry snapshots")
        self.scene = self.unwrapped.scene
        self.robot, self.red, self.blue = self.scene["robot"], self.scene["cube_2"], self.scene["cube_1"]
        self.arm_ids, _ = self.robot.find_joints(["panda_joint[1-7]"], preserve_order=True)
        if len(self.arm_ids) != 7:
            raise ValueError("expected seven Franka arm joints")
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.total_episode_steps = torch.zeros_like(self.steps)
        self.stage = torch.full_like(self.steps, int(self.initial_skill))
        self.stable_count, self.bad_count = torch.zeros_like(self.steps), torch.zeros_like(self.steps)
        self.prev_unit = torch.zeros(self.num_envs, 5, device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.last_source_idx = torch.zeros_like(self.steps)
        self.last_offsets = torch.zeros(self.num_envs, 2, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.release_passed = self.finished.clone()
        self.last_success, self.last_fail, self.last_timeout = [self.finished.clone() for _ in range(3)]
        self.last_handoff = self.finished.clone()
        self.last_skill_success = self.finished.clone()
        self.last_raw_action = torch.zeros(self.num_envs, 7, device=self.device)
        self.auto_reset = not chain
        self.measured = None
        self.total_steps = self.completed = self.successes = self.failures = self.timeouts = 0
        self.restore_calls = 0
        self.handoff_events = []
        self.unwrapped.single_action_space = gym.spaces.Box(-1., 1., (5,), dtype=np.float32)
        self.unwrapped.action_space = gym.vector.utils.batch_space(self.unwrapped.single_action_space, self.num_envs)
        self.unwrapped.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-10., 10., (46,), dtype=np.float32)})
        self.unwrapped.observation_space = gym.vector.utils.batch_space(self.unwrapped.single_observation_space, self.num_envs)

    def _restore(self, ids):
        if not len(ids):
            return
        choices = torch.randint(len(self.payloads), (len(ids),), generator=self.rng)
        offsets = (torch.rand(len(ids), 2, generator=self.rng) * 2 - 1) * self.xy_jitter_m
        for choice in choices.unique().tolist():
            mask = choices == choice
            selected = ids[mask.to(self.device)]
            payload = self.payloads[choice]
            state = expand_single_env_state(payload["scene_state"], len(selected), self.device)
            delta = offsets[mask].to(self.device)
            if self.initial_skill == Skill.RELEASE_STABILIZE:
                randomize_rigid_object_xy(state, "cube_1", delta)
            self.unwrapped.reset_to(state, env_ids=selected, is_relative=True)
            self.restore_calls += 1
            self.last_source_idx[selected] = choice
            self.last_offsets[selected] = delta
            self.prev_unit[selected] = torch.as_tensor(payload.get("previous_action", [0.] * 5), device=self.device)
        self.stage[ids] = int(self.initial_skill)
        for tensor in (self.steps, self.total_episode_steps, self.stable_count, self.bad_count, self.returns,
                       self.finished, self.release_passed, self.last_success, self.last_fail, self.last_timeout,
                       self.last_handoff, self.last_skill_success):
            tensor[ids] = 0
        self.unwrapped.episode_length_buf[ids] = 0
        m = self._measure()
        valid = entry_valid(self.initial_skill, {k: v[ids] for k, v in m.items()}, self.task_cfg)
        if not valid.all():
            raise RuntimeError(f"illegal restored {self.initial_skill.name} entries: {ids[~valid].tolist()}")

    def _obs(self):
        m, c = self.measured, self.task_cfg
        rel = (m["red"] - m["blue"]).clone()
        rel[:, 2] -= c.stack_height_m
        hold = torch.where(self.stage == int(Skill.RETREAT), c.retreat_hold_steps, c.stable_steps)
        obs = torch.cat([rel / .05, (m["ee"] - m["red"]) / .05,
                         m["vel"] / .2, m["angular"] / 2., m["red_quat"], m["blue_quat"],
                         m["grip"] / .04, m["force"] / 10., m["q"] / 3., m["qd"] / 2.,
                         self.prev_unit, (self.stable_count / hold).unsqueeze(1),
                         (self.bad_count / c.failure_consecutive_steps).unsqueeze(1),
                         (self.steps / c.episode_steps).unsqueeze(1)], dim=-1)
        if obs.shape != (self.num_envs, 46) or not torch.isfinite(obs).all():
            raise RuntimeError("invalid StagePPO observation")
        return TensorDict({"policy": obs.clamp(-10, 10)}, batch_size=[self.num_envs])

    def reset(self):
        self._restore(torch.arange(self.num_envs, device=self.device))
        self.measured = self._measure()
        self.handoff_events.clear()
        return self._obs(), {}

    def get_observations(self):
        return self.reset()[0] if self.measured is None else self._obs()

    def _handoff(self, ids):
        """Only Python bookkeeping. Never writes to the simulator or actuator."""
        before = self._measure()
        self.stage[ids] = int(Skill.RETREAT)
        self.steps[ids] = 0
        self.stable_count[ids] = 0
        self.bad_count[ids] = 0
        # Preserve previous action and all physical state at the skill boundary.
        after = self._measure()
        for key in before:
            torch.testing.assert_close(before[key][ids], after[key][ids], rtol=0, atol=0)
        for i in ids.tolist():
            self.handoff_events.append({"env": i, "step": int(self.total_episode_steps[i]),
                "from": "RELEASE_STABILIZE", "to": "RETREAT", "physical_measurements_max_abs_delta": 0.,
                "restore_calls": self.restore_calls, "previous_action": self.prev_unit[i].tolist(),
                "red_speed_mps": float(before["speed"][i]),
                "ee_red_distance_m": float((before["ee"][i] - before["red"][i]).norm())})

    def step(self, actions):
        if self.measured is None:
            self.reset()
        if self.chain and self.auto_reset:
            raise RuntimeError("chain evaluations may not auto-reset")
        c, before = self.task_cfg, self.measured
        active = ~self.finished
        stage_before = self.stage.clone()
        raw, unit = direct_action(torch.as_tensor(actions, device=self.device, dtype=torch.float32), c)
        self.env.step(raw)
        if self.unwrapped.reset_buf.any():
            raise RuntimeError("base environment reset during skill execution")
        self.last_raw_action.copy_(raw)
        self.prev_unit.copy_(unit)
        self.steps += active.long()
        self.total_episode_steps += active.long()
        m = self._measure()
        success, failure, timeout = [torch.zeros_like(self.finished) for _ in range(3)]
        reward = torch.zeros(self.num_envs, device=self.device)
        for skill in Skill:
            mask = (stage_before == int(skill)) & active
            if not mask.any():
                continue
            old = {k: v[mask] for k, v in before.items()}
            new = {k: v[mask] for k, v in m.items()}
            s, f, t, stable, bad, _ = skill_flags(skill, new, self.stable_count[mask], self.bad_count[mask], self.steps[mask], c)
            success[mask], failure[mask], timeout[mask] = s, f, t
            self.stable_count[mask], self.bad_count[mask] = stable, bad
            reward[mask], _ = direct_reward(skill_potential(skill, old, c), skill_potential(skill, new, c), unit[mask], s, f, t, c)
        self.last_skill_success.copy_(success)
        self.release_passed |= success & (stage_before == int(Skill.RELEASE_STABILIZE))
        if self.chain:
            _, handoff, done = chain_transition(stage_before, success, failure, timeout)
            final_success = success & (stage_before == int(Skill.RETREAT))
            if handoff.any():
                self._handoff(handoff.nonzero(as_tuple=False).flatten())
        else:
            handoff = torch.zeros_like(success)
            done, final_success = success | failure | timeout, success
        self.last_handoff.copy_(handoff)
        self.last_success |= final_success
        self.last_fail |= failure
        self.last_timeout |= timeout
        self.finished |= done
        self.returns += reward
        self.measured = m
        extras = {"log": {"stageppo/reward": reward.mean(),
                          "stageppo/open_command": (raw[:, 6] > 0).float().mean()}}
        ids = done.nonzero(as_tuple=False).flatten()
        if self.auto_reset:
            self.total_steps += self.num_envs
            self.completed += len(ids)
            self.successes += int(final_success.sum())
            self.failures += int(failure.sum())
            self.timeouts += int(timeout.sum())
            if len(ids):
                extras["log"].update({"stageppo/episode_return": self.returns[ids].mean(),
                                      "stageppo/episode_success": final_success[ids].float().mean()})
                self._restore(ids)
                self.measured = self._measure()
        # Independent skill PPO uses true finite-horizon terminals. Chain mode
        # is evaluation-only: upstream completion never ends the whole task.
        return self._obs(), reward, done, extras

    def statistics(self):
        return {"skill": self.initial_skill.name, "transitions": self.total_steps,
                "completed_episodes": self.completed, "successes": self.successes,
                "physical_failures": self.failures, "timeouts": self.timeouts,
                "training_episode_success_rate": self.successes / self.completed if self.completed else None,
                "contract": asdict(self.task_cfg)}
