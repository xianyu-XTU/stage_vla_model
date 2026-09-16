"""M19 frozen-A1 recovery-only PPO vector environment.

The validated A1_RF150 movement actor and M16 PlaceBC are frozen. PPO learns
only a small 4-D adapter correction while the gripper is forced CLOSED. D2
uses the historical five TRAIN live handoffs; D3 optionally uses a diverse
50..100-snapshot TRAIN-only bank from a disjoint seed range. The control,
reward, ready envelope and network contract are otherwise identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from stage_vla.envs.state_readers import (
    net_force_per_env,
    read_grasp_state,
    read_placement_state,
    resolve_frame_indices,
)
from stage_vla.integration.handoff_adaptation import (
    HANDOFF_ROLE,
    TRAIN_SEEDS,
    discover_handoff_snapshots,
    validate_d3_snapshot_seed_set,
    validate_d3_train_seed,
    validate_train_seed,
)
from stage_vla.integration.handoff_recovery import (
    RECOVERY_ACTION_DIM,
    RecoveryReadyConfig,
    RecoveryReadyTracker,
    RecoveryRewardConfig,
    compose_recovery_action,
    ready_now,
    recovery_reward,
)
from stage_vla.lightweight_vla.geometry_phase import ALIGN
from stage_vla.lightweight_vla.place_bc import PlaceBC
from stage_vla.lightweight_vla.residual_place_core import (
    CONT_DIM,
    RESIDUAL_OBS_DIM,
    compose_residual_cont_action,
    residual_observation_from_bc_state,
)
from stage_vla.rl.hard_case_curriculum import sample_uniform_xy
from stage_vla.rl.learned_release import FrozenA1MovementPolicy
from stage_vla.rl.place_snapshot import expand_single_env_state, load_place_snapshot, randomize_rigid_object_xy
from stage_vla.stages import PhysicalGraspConfig, physical_grasp_diagnostics

STRICT_STACK_HEIGHT_DIFF_M = 0.0468
TARGET_OFFSET = torch.tensor([0.0, 0.0, 0.04], dtype=torch.float32)


@dataclass
class _RecoveryMeasurements:
    red_pos: torch.Tensor
    blue_pos: torch.Tensor
    red_lin_vel: torch.Tensor
    red_ang_vel: torch.Tensor
    gripper_joint_pos: torch.Tensor
    gripper_open: torch.Tensor
    ee_pos: torch.Tensor
    finger_forces: torch.Tensor
    physical_grasp: torch.Tensor
    red_speed: torch.Tensor
    strict_z_error: torch.Tensor


class HandoffRecoveryVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper for the isolated M19-D2 transition adapter."""

    def __init__(
        self,
        env,
        *,
        handoff_snapshot_dir: str | Path,
        bc_checkpoint: str | Path,
        movement_checkpoint: str | Path,
        a1_residual_scales: torch.Tensor,
        ready_cfg: RecoveryReadyConfig,
        reward_cfg: RecoveryRewardConfig | None = None,
        adapter_scale: torch.Tensor | tuple[float, float, float, float] = (0.008, 0.008, 0.006, 0.012),
        xy_jitter_m: float = 0.002,
        episode_steps: int = 80,
        clip_actions: float = 1.0,
        seed: int = 17,
        training_resets: bool = True,
        snapshot_bank_mode: str = "legacy5",
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.num_actions = RECOVERY_ACTION_DIM
        self.recovery_observation_dim = RESIDUAL_OBS_DIM
        self.ready_cfg = ready_cfg
        self.ready_cfg.validate()
        self.reward_cfg = reward_cfg or RecoveryRewardConfig()
        self.xy_jitter_m = float(xy_jitter_m)
        self.episode_steps = int(episode_steps)
        self.training_resets = bool(training_resets)
        self.snapshot_bank_mode = str(snapshot_bank_mode).strip().lower()
        if self.snapshot_bank_mode not in {"legacy5", "d3_diverse"}:
            raise ValueError("snapshot_bank_mode must be 'legacy5' or 'd3_diverse'")
        if self.xy_jitter_m < 0:
            raise ValueError("xy_jitter_m must be >= 0")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be > 0")

        self._rng = torch.Generator(device="cpu").manual_seed(int(seed))
        self.adapter_scale = torch.as_tensor(adapter_scale, dtype=torch.float32, device=self.device)
        if self.adapter_scale.shape != (4,):
            raise ValueError("adapter_scale must be [4]")
        if torch.any(self.adapter_scale <= 0):
            raise ValueError("adapter scales must be positive")

        self.handoff_snapshot_dir = str(Path(handoff_snapshot_dir))
        self._handoff_payloads: list[dict] = []
        for snap_path in discover_handoff_snapshots(handoff_snapshot_dir):
            payload = load_place_snapshot(snap_path)
            if str(payload.get("snapshot_role", "")).lower() != HANDOFF_ROLE:
                continue
            raw_seed = int(payload.get("source_seed", -1))
            seed_i = (
                validate_train_seed(raw_seed)
                if self.snapshot_bank_mode == "legacy5"
                else validate_d3_train_seed(raw_seed)
            )
            payload = dict(payload)
            payload["_source_seed"] = seed_i
            payload["_path"] = str(snap_path)
            self._handoff_payloads.append(payload)

        bank_seeds = tuple(int(p["_source_seed"]) for p in self._handoff_payloads)
        if self.snapshot_bank_mode == "legacy5":
            if len(self._handoff_payloads) != 5:
                raise ValueError(
                    f"M19-D2 requires exactly five TRAIN live-handoff snapshots, got {len(self._handoff_payloads)}"
                )
            if set(bank_seeds) != set(TRAIN_SEEDS):
                raise ValueError("M19-D2 snapshot bank must contain exactly the five TRAIN seeds")
        else:
            validate_d3_snapshot_seed_set(bank_seeds)
        self.snapshot_seeds = bank_seeds

        self.a1_residual_scales = torch.as_tensor(a1_residual_scales, dtype=torch.float32, device=self.device)
        if self.a1_residual_scales.shape != (5, CONT_DIM):
            raise ValueError("a1_residual_scales must be [5,4]")

        ck = torch.load(str(Path(bc_checkpoint)), map_location=self.device, weights_only=False)
        self.bc = PlaceBC().to(self.device).eval()
        self.bc.load_state_dict(ck["state_dict"])
        self.bc.requires_grad_(False)
        self.bc_checkpoint = str(Path(bc_checkpoint))

        self.a1 = FrozenA1MovementPolicy(movement_checkpoint, self.device)
        self.movement_checkpoint = str(Path(movement_checkpoint))
        if any(p.requires_grad for p in self.a1.parameters()):
            raise RuntimeError("A1 must be fully frozen in M19-D2")

        self._scene = self.unwrapped.scene
        self._ee_frame = self._scene["ee_frame"]
        self._frame_indices = resolve_frame_indices(self._ee_frame.data.target_frame_names)
        self._physical_cfg = PhysicalGraspConfig(
            radial_tolerance_m=0.03,
            height_tolerance_m=0.012,
            contact_force_threshold_n=0.5,
            endpoint_margin=0.05,
        )
        self._ready_tracker = RecoveryReadyTracker(self.num_envs, self.device, self.ready_cfg)
        self._prev_cont = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._grasp_grace = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_step_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_step_lost_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_step_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cached_state: torch.Tensor | None = None
        self._cached_m: _RecoveryMeasurements | None = None
        self._reset_seed = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._reset_counts = torch.zeros(len(self._handoff_payloads), dtype=torch.long, device=self.device)

        import gymnasium as gym
        self.unwrapped.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.unwrapped.action_space = gym.vector.utils.batch_space(self.unwrapped.single_action_space, self.num_envs)

    def _all_ids(self) -> torch.Tensor:
        return torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    def _restore_ids(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        picks = torch.randint(0, len(self._handoff_payloads), (len(env_ids),), generator=self._rng)
        for idx in torch.unique(picks).tolist():
            mask_cpu = picks == int(idx)
            ids = env_ids[mask_cpu.to(self.device)]
            payload = self._handoff_payloads[int(idx)]
            state = expand_single_env_state(payload["scene_state"], len(ids), self.device)
            if self.training_resets and self.xy_jitter_m > 0:
                offsets = sample_uniform_xy(len(ids), self.xy_jitter_m, self._rng).to(self.device)
                randomize_rigid_object_xy(state, "cube_1", offsets)
            self.unwrapped.reset_to(state, env_ids=ids, is_relative=True)
            self._reset_seed[ids] = int(payload["_source_seed"])
            self._reset_counts[int(idx)] += len(ids)

        self._prev_cont[env_ids] = 0.0
        self._episode_step[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._grasp_grace[env_ids] = 2
        self._ready_tracker.reset(env_ids)
        self.last_step_ready[env_ids] = False
        self.last_step_lost_grasp[env_ids] = False
        self.last_step_timeout[env_ids] = False
        if hasattr(self.unwrapped, "episode_length_buf"):
            self.unwrapped.episode_length_buf[env_ids] = 0

    def _measure(self) -> _RecoveryMeasurements:
        p = read_placement_state(self.unwrapped, gripper_open_tolerance_m=0.002)
        g = read_grasp_state(self.unwrapped)
        f_l = net_force_per_env(self._scene.sensors["left_finger_contact"])
        f_r = net_force_per_env(self._scene.sensors["right_finger_contact"])
        physical = physical_grasp_diagnostics(
            g.red_pos_w,
            g.left_tip_w,
            g.right_tip_w,
            g.finger_a_force_n,
            g.finger_b_force_n,
            cfg=self._physical_cfg,
        ).physical_grasp
        ee_pos = self._ee_frame.data.target_pos_w[:, self._frame_indices.end_effector]
        red_speed = torch.linalg.vector_norm(p.red_lin_vel_w, dim=-1)
        strict_z = p.red_pos_w[:, 2] - (p.blue_pos_w[:, 2] + STRICT_STACK_HEIGHT_DIFF_M)
        return _RecoveryMeasurements(
            red_pos=p.red_pos_w,
            blue_pos=p.blue_pos_w,
            red_lin_vel=p.red_lin_vel_w,
            red_ang_vel=p.red_ang_vel_w,
            gripper_joint_pos=p.gripper_joint_pos,
            gripper_open=p.gripper_open,
            ee_pos=ee_pos,
            finger_forces=torch.stack([f_l, f_r], dim=-1),
            physical_grasp=physical,
            red_speed=red_speed,
            strict_z_error=strict_z,
        )

    def _build_state(self, m: _RecoveryMeasurements) -> torch.Tensor:
        target = m.blue_pos + TARGET_OFFSET.to(self.device).reshape(1, 3)
        phase = torch.full((self.num_envs,), ALIGN, dtype=torch.long, device=self.device)
        micro = torch.nn.functional.one_hot(phase, num_classes=5).to(torch.float32)
        state = torch.cat(
            [
                m.red_pos - target,
                m.ee_pos - m.red_pos,
                m.red_lin_vel,
                m.red_ang_vel,
                m.gripper_joint_pos,
                m.finger_forces,
                self._prev_cont,
                micro,
            ],
            dim=-1,
        ).to(torch.float32)
        if state.shape[-1] != 25:
            raise RuntimeError(f"M19-D2 state invariant broken: {tuple(state.shape)}")
        return state

    def _refresh(self) -> None:
        self._cached_m = self._measure()
        self._cached_state = self._build_state(self._cached_m)

    def _obs(self) -> TensorDict:
        if self._cached_state is None:
            raise RuntimeError("recovery wrapper state not initialized")
        obs = residual_observation_from_bc_state(self._cached_state)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def get_observations(self):
        if self._cached_state is None:
            self._refresh()
        return self._obs()

    def reset(self):
        self._restore_ids(self._all_ids())
        self._refresh()
        return self._obs(), {}

    def diagnostics(self) -> dict[str, float | int]:
        if self._cached_m is None:
            self._refresh()
        m = self._cached_m
        xy = torch.linalg.vector_norm(m.red_pos[:, :2] - m.blue_pos[:, :2], dim=-1)
        z_abs = (m.strict_z_error - float(self.ready_cfg.strict_z_center_m)).abs()
        return {
            "snapshot_count": int(len(self._handoff_payloads)),
            "snapshot_bank_mode": self.snapshot_bank_mode,
            "unique_snapshot_seed_count": int(len(set(self.snapshot_seeds))),
            "xy_mean_m": float(xy.mean().item()),
            "red_speed_mean_mps": float(m.red_speed.mean().item()),
            "entry_z_abs_mean_m": float(z_abs.mean().item()),
            "physical_grasp_frac": float(m.physical_grasp.float().mean().item()),
            "a1_requires_grad_count": int(sum(int(p.requires_grad) for p in self.a1.parameters())),
            "ready_xy_max_m": float(self.ready_cfg.xy_max_m),
            "ready_red_speed_max_mps": float(self.ready_cfg.red_speed_max_mps),
            "ready_z_center_m": float(self.ready_cfg.strict_z_center_m),
            "ready_z_tolerance_m": float(self.ready_cfg.strict_z_tolerance_m),
            "ready_consecutive_steps": int(self.ready_cfg.consecutive_steps),
        }

    def step(self, actions):
        if self._cached_state is None or self._cached_m is None:
            self.reset()
        adapter_unit = torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-1.0, 1.0)
        if adapter_unit.shape != (self.num_envs, 4):
            raise ValueError(f"M19-D2 adapter action expected {(self.num_envs, 4)}, got {tuple(adapter_unit.shape)}")
        prev_state = self._cached_state.clone()
        prev_m = self._cached_m
        align_phase = torch.full((self.num_envs,), ALIGN, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            base_cont, _ = self.bc.act(prev_state)
            a1_unit = self.a1(residual_observation_from_bc_state(prev_state)).clamp(-1.0, 1.0)
            frozen_a1_cont, _ = compose_residual_cont_action(
                base_cont, a1_unit, align_phase, self.a1_residual_scales
            )
        final_cont, adapter_delta = compose_recovery_action(
            frozen_a1_cont,
            adapter_unit,
            self.adapter_scale,
        )
        grip_close = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        raw = torch.stack(
            [
                final_cont[:, 0], final_cont[:, 1], final_cont[:, 2],
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                final_cont[:, 3], grip_close,
            ],
            dim=-1,
        )
        _, _, _, _, base_extras = self.env.step(raw)
        self._prev_cont.copy_(final_cont.detach())
        self._episode_step += 1
        self._refresh()
        next_state = self._cached_state
        next_m = self._cached_m

        prev_xy = torch.linalg.vector_norm(prev_m.red_pos[:, :2] - prev_m.blue_pos[:, :2], dim=-1)
        next_xy = torch.linalg.vector_norm(next_m.red_pos[:, :2] - next_m.blue_pos[:, :2], dim=-1)
        prev_z_abs = (prev_m.strict_z_error - float(self.ready_cfg.strict_z_center_m)).abs()
        next_z_abs = (next_m.strict_z_error - float(self.ready_cfg.strict_z_center_m)).abs()

        now = ready_now(
            xy_error_m=next_xy,
            strict_z_error_m=next_m.strict_z_error,
            red_speed_mps=next_m.red_speed,
            physical_grasp=next_m.physical_grasp,
            gripper_open=next_m.gripper_open,
            cfg=self.ready_cfg,
        )
        ready, newly_ready = self._ready_tracker.update(now)
        local_red_z = next_m.red_pos[:, 2] - self._scene.env_origins[:, 2]
        grace_active = self._grasp_grace > 0
        lost_grasp = (~next_m.physical_grasp & ~grace_active) | (local_red_z < 0.045) | next_m.gripper_open
        self._grasp_grace = torch.clamp(self._grasp_grace - 1, min=0)
        timeout = self._episode_step >= self.episode_steps
        dones = ready | lost_grasp | timeout

        reward = recovery_reward(
            prev_xy=prev_xy,
            next_xy=next_xy,
            prev_speed=prev_m.red_speed,
            next_speed=next_m.red_speed,
            prev_z_abs=prev_z_abs,
            next_z_abs=next_z_abs,
            adapter_unit=adapter_unit,
            ready=newly_ready,
            lost_grasp=lost_grasp,
            timeout=timeout & ~ready & ~lost_grasp,
            ready_cfg=self.ready_cfg,
            reward_cfg=self.reward_cfg,
        )
        self._episode_return += reward
        self.last_step_ready.copy_(ready)
        self.last_step_lost_grasp.copy_(lost_grasp)
        self.last_step_timeout.copy_(timeout & ~ready & ~lost_grasp)

        extras = dict(base_extras) if isinstance(base_extras, dict) else {}
        extras.setdefault("log", {})
        extras["log"].update(
            {
                "m19d2/xy_error_m": next_xy.mean(),
                "m19d2/red_speed_mps": next_m.red_speed.mean(),
                "m19d2/entry_z_abs_m": next_z_abs.mean(),
                "m19d2/adapter_abs_mean": adapter_delta.abs().mean(),
                "m19d2/ready_now_fraction": now.float().mean(),
                "m19d2/ready_latched_fraction": ready.float().mean(),
                "m19d2/lost_grasp_fraction": lost_grasp.float().mean(),
                "m19d2/physical_grasp_fraction": next_m.physical_grasp.float().mean(),
            }
        )

        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if len(done_ids) > 0:
            extras["log"]["m19d2/episode_return_done"] = self._episode_return[done_ids].mean()
            extras["log"]["m19d2/ready_done"] = ready[done_ids].float().mean()
            extras["log"]["m19d2/lost_grasp_done"] = lost_grasp[done_ids].float().mean()
            extras["log"]["m19d2/episode_length_done"] = self._episode_step[done_ids].float().mean()
            if self.training_resets:
                self._restore_ids(done_ids)
                self._refresh()

        return self._obs(), reward, dones, extras
