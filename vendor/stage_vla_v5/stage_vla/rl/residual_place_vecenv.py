"""M17 PLACE-only Residual PPO vector environment.

Design goals
------------
* Freeze the validated M16 PlaceBC as the *base* controller.
* PPO normally predicts a bounded 4-D residual; M18 optionally adds a fifth constrained RELEASE signal.
* The PPO observation is the existing 25-D object-centric BC state plus six
  explicit terminal-error features (31-D total).
* Gripper timing follows the causal GeometryPhaseTracker contract. M17-A5 can
  additionally delay OPEN inside RELEASE until strict geometry is low-speed for
  N consecutive frames. A6-R2 can instead use an injected exact red<->blue
  support-contact provider. RELEASE/FAIL base motion is locked to zero except
  for the bounded A6 +Z retreat primitive.
* Episodes start from a validated relative PLACE snapshot captured in a separate
  1-env process (red cube already grasped and transported above blue).  Per episode only the blue target XY is randomized,
  which changes the required placement displacement without breaking the grasp.
* Individual finished envs are restored with Isaac Lab ``reset_to(...,
  is_relative=True)``; raw trace replay never occurs in the vector training scene.

The wrapper deliberately bypasses the base task reward/termination managers for
M17 training.  The base task is used only as a physics/action backend.
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
from stage_vla.lightweight_vla.geometry_phase import (
    ALIGN,
    FAIL,
    RELEASE,
    GeometryPhaseConfig,
    GeometryPhaseTracker,
)
from stage_vla.lightweight_vla.place_bc import PlaceBC
from stage_vla.lightweight_vla.residual_place_core import (
    CONT_DIM,
    RESIDUAL_OBS_DIM,
    STRICT_Z_BIAS_M,
    ResidualRewardConfig,
    compose_residual_cont_action,
    residual_observation_from_bc_state,
    residual_place_reward,
    strict_grip_command,
)
from stage_vla.rl.place_snapshot import expand_single_env_state, load_place_snapshot, randomize_rigid_object_xy
from stage_vla.rl.learned_release import (
    LEARNED_RELEASE_ACTION_DIM,
    FrozenA1MovementPolicy,
    LearnedReleaseConfig,
    LearnedReleaseController,
    apply_learned_release_reward_contract,
)
from stage_vla.rl.terminal_stability import ReleaseStabilityConfig, ReleaseStabilityGate
from stage_vla.rl.release_primitive import (
    BASELINE as RELEASE_PRIMITIVE_BASELINE,
    SUPPORT_RETREAT,
    ReleasePrimitiveConfig,
    ReleasePrimitiveController,
)
from stage_vla.integration.handoff_adaptation import (
    HANDOFF_ROLE,
    TRAIN_SEEDS as HANDOFF_TRAIN_SEEDS,
    discover_handoff_snapshots,
    sample_handoff_mask,
    validate_train_seed as validate_handoff_train_seed,
)
from stage_vla.rl.hard_case_curriculum import (
    EDGE,
    NORMAL,
    TERMINAL,
    HardCaseCurriculumConfig,
    assign_reset_modes,
    discover_terminal_snapshots,
    sample_edge_xy,
    sample_uniform_xy,
)
from stage_vla.stages import PhysicalGraspConfig, physical_grasp_diagnostics


CUBE_SIZE_M = 0.04
TARGET_HEIGHT_DIFF_M = 0.0468
TARGET_OFFSET = torch.tensor([0.0, 0.0, CUBE_SIZE_M], dtype=torch.float32)


@dataclass
class _Measurements:
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
    support_force_n: torch.Tensor


class ResidualPlaceVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL VecEnv that executes frozen BC + bounded residual PPO."""

    def __init__(
        self,
        env,
        *,
        bc_checkpoint: str | Path,
        place_snapshot: str | Path,
        residual_scales: torch.Tensor,
        clip_actions: float = 1.0,
        target_xy_range_m: float = 0.03,
        episode_steps: int = 180,
        stable_steps: int = 5,
        seed: int = 0,
        reward_cfg: ResidualRewardConfig | None = None,
        training_resets: bool = True,
        curriculum_snapshot_dir: str | Path | None = None,
        hard_case_fraction: float = 0.0,
        terminal_fraction_within_hard: float = 0.5,
        hard_edge_min_fraction: float = 0.8,
        terminal_xy_jitter_m: float = 0.005,
        handoff_snapshot_dir: str | Path | None = None,
        handoff_fraction: float = 0.0,
        handoff_xy_jitter_m: float = 0.005,
        release_gate_steps: int = 0,
        release_speed_threshold_mps: float = 0.02,
        release_primitive_mode: str = "baseline",
        release_open_hold_steps: int = 3,
        release_retreat_steps: int = 8,
        release_retreat_dz_per_step: float = 0.003,
        support_gate_steps: int = 2,
        support_force_threshold_n: float = 0.20,
        support_max_wait_steps: int = 20,
        support_contact_provider=None,
        learn_release_timing: bool = False,
        frozen_movement_checkpoint: str | Path | None = None,
        release_signal_threshold: float = 0.0,
        release_wait_free_steps: int = 2,
        release_wait_penalty: float = 0.004,
        invalid_release_request_penalty: float = 0.002,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.learn_release_timing = bool(learn_release_timing)
        self.num_actions = LEARNED_RELEASE_ACTION_DIM if self.learn_release_timing else CONT_DIM
        self.residual_observation_dim = RESIDUAL_OBS_DIM
        self.target_xy_range_m = float(target_xy_range_m)
        self.place_episode_steps = int(episode_steps)
        self.stable_steps = int(stable_steps)
        self.training_resets = bool(training_resets)
        if self.target_xy_range_m < 0:
            raise ValueError("target_xy_range_m must be >= 0")
        if self.place_episode_steps <= 0:
            raise ValueError("episode_steps must be > 0")
        if self.stable_steps <= 0:
            raise ValueError("stable_steps must be > 0")

        self._rng = torch.Generator(device="cpu").manual_seed(int(seed))
        snapshot_payload = load_place_snapshot(place_snapshot)
        self.place_snapshot = str(Path(place_snapshot))
        self.snapshot_metadata = {k: v for k, v in snapshot_payload.items() if k != "scene_state"}
        self._snapshot_state_single = snapshot_payload["scene_state"]
        self._snapshot_initial_phase = int(snapshot_payload.get("initial_phase", ALIGN))

        self._curriculum_cfg = HardCaseCurriculumConfig(
            hard_fraction=float(hard_case_fraction),
            terminal_fraction_within_hard=float(terminal_fraction_within_hard),
            edge_min_fraction=float(hard_edge_min_fraction),
            terminal_xy_jitter_m=float(terminal_xy_jitter_m),
        )
        self._curriculum_cfg.validate()
        self._terminal_snapshot_payloads: list[dict] = []
        self.curriculum_snapshot_dir = None
        if curriculum_snapshot_dir is not None:
            self.curriculum_snapshot_dir = str(Path(curriculum_snapshot_dir))
            for snap_path in discover_terminal_snapshots(curriculum_snapshot_dir):
                payload = load_place_snapshot(snap_path)
                role = str(payload.get("snapshot_role", "")).lower()
                phase = int(payload.get("initial_phase", ALIGN))
                if role not in {"descend", "settle"} or phase not in {1, 2}:
                    continue
                # Never allow held-out unseen snapshots to enter the bank. The
                # capture tool writes the source seed into every payload.
                seed = int(payload.get("source_seed", -1))
                if seed not in {1004, 1009, 1015, 1020, 1031}:
                    raise ValueError(
                        f"A4 curriculum snapshot {snap_path} has non-training seed {seed}; "
                        "held-out unseen states must not be used for training"
                    )
                payload = dict(payload)
                payload["_path"] = str(snap_path)
                self._terminal_snapshot_payloads.append(payload)
        if self._curriculum_cfg.hard_fraction > 0 and (
            self._curriculum_cfg.terminal_fraction_within_hard > 0
            and not self._terminal_snapshot_payloads
        ):
            raise ValueError(
                "terminal hard-case resets requested but no TRAIN DESCEND/SETTLE snapshots were found; "
                "run tools/m17_capture_curriculum_snapshots.py first"
            )

        # M19-D1: adapt the validated A1 policy to the real TRANSPORT->PLACE
        # entry distribution without leaking held-out seeds.  This bank is
        # independent of the older A4 hard-case curriculum; mixing both in one
        # experiment would confound attribution, so the wrapper rejects it.
        self.handoff_fraction = float(handoff_fraction)
        self.handoff_xy_jitter_m = float(handoff_xy_jitter_m)
        if not 0.0 <= self.handoff_fraction <= 1.0:
            raise ValueError("handoff_fraction must be in [0,1]")
        if self.handoff_xy_jitter_m < 0.0:
            raise ValueError("handoff_xy_jitter_m must be >= 0")
        if self.handoff_fraction > 0.0 and self._curriculum_cfg.hard_fraction > 0.0:
            raise ValueError(
                "M19-D1 handoff adaptation and M17-A4 hard-case curriculum cannot be enabled together; "
                "change one reset-distribution variable at a time"
            )
        self.handoff_snapshot_dir = None
        self._handoff_snapshot_payloads: list[dict] = []
        if handoff_snapshot_dir is not None:
            self.handoff_snapshot_dir = str(Path(handoff_snapshot_dir))
            for snap_path in discover_handoff_snapshots(handoff_snapshot_dir):
                payload = load_place_snapshot(snap_path)
                seed = validate_handoff_train_seed(int(payload.get("source_seed", -1)))
                role = str(payload.get("snapshot_role", "")).lower()
                if role != HANDOFF_ROLE:
                    continue
                payload = dict(payload)
                payload["_path"] = str(snap_path)
                payload["_source_seed"] = seed
                self._handoff_snapshot_payloads.append(payload)
        if self.handoff_fraction > 0.0 and not self._handoff_snapshot_payloads:
            raise ValueError(
                "handoff adaptation requested but no TRAIN live-handoff snapshots were found; "
                "run tools/m19_capture_handoff_snapshots.py first"
            )

        self.residual_scales = torch.as_tensor(
            residual_scales, dtype=torch.float32, device=self.device
        )
        if self.residual_scales.shape != (5, CONT_DIM):
            raise ValueError("residual_scales must have shape [5,4]")

        # Frozen M16 base policy.
        ck = torch.load(str(Path(bc_checkpoint)), map_location=self.device, weights_only=False)
        self.bc = PlaceBC().to(self.device).eval()
        self.bc.load_state_dict(ck["state_dict"])
        self.bc.requires_grad_(False)
        self.bc_checkpoint = str(Path(bc_checkpoint))

        scene = self.unwrapped.scene
        self._scene = scene
        self._robot = scene["robot"]
        self._red = scene.rigid_objects["cube_2"]
        self._blue = scene.rigid_objects["cube_1"]
        self._green = scene.rigid_objects.get("cube_3", None)
        self._ee_frame = scene["ee_frame"]
        self._frame_indices = resolve_frame_indices(self._ee_frame.data.target_frame_names)

        self._physical_cfg = PhysicalGraspConfig(
            radial_tolerance_m=0.03,
            height_tolerance_m=0.012,
            contact_force_threshold_n=0.5,
            endpoint_margin=0.05,
        )
        self._phase_tracker = GeometryPhaseTracker(self.num_envs, GeometryPhaseConfig())
        self._release_gate_cfg = ReleaseStabilityConfig(
            gate_steps=int(release_gate_steps),
            speed_threshold_mps=float(release_speed_threshold_mps),
        )
        self._release_gate = ReleaseStabilityGate(self.num_envs, self.device, self._release_gate_cfg)
        self._release_primitive_cfg = ReleasePrimitiveConfig(
            mode=str(release_primitive_mode),
            open_hold_steps=int(release_open_hold_steps),
            retreat_steps=int(release_retreat_steps),
            retreat_dz_per_step=float(release_retreat_dz_per_step),
            support_gate_steps=int(support_gate_steps),
            support_force_threshold_n=float(support_force_threshold_n),
            support_max_wait_steps=int(support_max_wait_steps),
        )
        self._release_primitive_cfg.validate()
        self._support_contact_provider = support_contact_provider
        if self._release_primitive_cfg.mode != RELEASE_PRIMITIVE_BASELINE and self._release_gate_cfg.gate_steps > 0:
            raise ValueError(
                "A5 low-speed release gate and A6 release primitive are mutually exclusive; "
                "set release_gate_steps=0 for A6 R1/R2"
            )
        if (
            self._release_primitive_cfg.mode == SUPPORT_RETREAT
            and self._support_contact_provider is None
            and "red_blue_support_contact" not in scene.sensors
        ):
            raise ValueError(
                "support_retreat requires a support-contact provider. For A6-R2 use "
                "Env0RawPairSupportProvider in 1-env evaluation; the legacy filtered "
                "ContactSensor backend is probe-only on this runtime."
            )
        self._release_primitive = ReleasePrimitiveController(
            self.num_envs, self.device, self._release_primitive_cfg
        )
        self._learned_release_cfg = LearnedReleaseConfig(
            threshold=float(release_signal_threshold),
            wait_free_steps=int(release_wait_free_steps),
            wait_penalty=float(release_wait_penalty),
            invalid_request_penalty=float(invalid_release_request_penalty),
        )
        self._learned_release = LearnedReleaseController(
            self.num_envs, self.device, self._learned_release_cfg
        )
        self._frozen_movement_policy = None
        if self.learn_release_timing:
            if frozen_movement_checkpoint is None:
                raise ValueError("learn_release_timing requires frozen_movement_checkpoint=A1_RF150")
            self._frozen_movement_policy = FrozenA1MovementPolicy(
                frozen_movement_checkpoint, self.device
            )
            self.frozen_movement_checkpoint = str(Path(frozen_movement_checkpoint))
            if self._release_gate_cfg.gate_steps > 0:
                raise ValueError("learned RELEASE timing cannot be combined with A5 low-speed gate")
            if self._release_primitive_cfg.mode != RELEASE_PRIMITIVE_BASELINE:
                raise ValueError("learned RELEASE timing cannot be combined with A6 manual release primitives")
        self._reward_cfg = reward_cfg or ResidualRewardConfig()

        self._prev_cont = torch.zeros(self.num_envs, CONT_DIM, device=self.device)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stable_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_step_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_step_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_step_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._grasp_grace = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_reset_mode = torch.full((self.num_envs,), NORMAL, dtype=torch.long, device=self.device)
        self._reset_mode_counts = torch.zeros(3, dtype=torch.long, device=self.device)
        self._last_reset_handoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_source_counts = torch.zeros(2, dtype=torch.long, device=self.device)  # canonical, live_handoff
        self._cached_state: torch.Tensor | None = None
        self._cached_phase: torch.Tensor | None = None
        self._cached_measurements: _Measurements | None = None

        # Keep public action-space metadata consistent with the residual actor.
        import gymnasium as gym

        self.unwrapped.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.unwrapped.action_space = gym.vector.utils.batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )

    # ------------------------------------------------------------------
    # Snapshot / reset handling
    # ------------------------------------------------------------------
    def _all_ids(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    def _sample_xy_offsets(self, n: int) -> torch.Tensor:
        return sample_uniform_xy(n, self.target_xy_range_m, self._rng).to(self.device)

    def _sample_edge_offsets(self, n: int) -> torch.Tensor:
        return sample_edge_xy(
            n, self.target_xy_range_m, self._rng, min_fraction=self._curriculum_cfg.edge_min_fraction
        ).to(self.device)

    def _sample_terminal_offsets(self, n: int) -> torch.Tensor:
        return sample_uniform_xy(
            n, self._curriculum_cfg.terminal_xy_jitter_m, self._rng
        ).to(self.device)

    def _restore_payload(
        self,
        env_ids: torch.Tensor,
        payload: dict,
        *,
        xy_offsets: torch.Tensor | None,
        reset_mode: int,
        handoff_source: bool = False,
    ) -> None:
        if len(env_ids) == 0:
            return
        state = expand_single_env_state(payload["scene_state"], len(env_ids), self.device)
        if xy_offsets is not None:
            randomize_rigid_object_xy(state, "cube_1", xy_offsets)
        self.unwrapped.reset_to(state, env_ids=env_ids, is_relative=True)
        initial_phase = int(payload.get("initial_phase", ALIGN))
        self._phase_tracker.set_phase(env_ids, initial_phase)
        self._last_reset_mode[env_ids] = int(reset_mode)
        self._reset_mode_counts[int(reset_mode)] += len(env_ids)
        self._last_reset_handoff[env_ids] = bool(handoff_source)
        self._reset_source_counts[1 if handoff_source else 0] += len(env_ids)

        self._prev_cont[env_ids] = 0.0
        self._episode_step[env_ids] = 0
        self._stable_count[env_ids] = 0
        self._release_gate.reset(env_ids)
        self._release_primitive.reset(env_ids)
        self._learned_release.reset(env_ids)
        self._episode_return[env_ids] = 0.0
        self._episode_success[env_ids] = False
        self._grasp_grace[env_ids] = 2
        if hasattr(self.unwrapped, "episode_length_buf"):
            self.unwrapped.episode_length_buf[env_ids] = 0

    def _restore_place(self, env_ids: torch.Tensor, *, randomize_target: bool) -> None:
        """Restore normal + failure-driven curriculum states into selected clones.

        NORMAL: exact M17 distribution (seed-1004 PLACE entry + uniform +/-3cm).
        EDGE: same snapshot, but at least one XY coordinate lies near the edge
              of the existing +/-3cm box; this targets recovery failures without
              widening the global distribution.
        TERMINAL: DESCEND/SETTLE snapshots captured only from the five TRAIN
                  seeds, with a small +/-5mm target jitter.
        """
        if len(env_ids) == 0:
            return

        # M19-D1 mixed reset distribution. Canonical resets preserve the exact
        # A1 training distribution; live-handoff resets restore one of the five
        # TRAIN-only TRANSPORT->PLACE states and add only a small target jitter.
        if randomize_target and self.handoff_fraction > 0.0:
            handoff_cpu = sample_handoff_mask(len(env_ids), self.handoff_fraction, self._rng)
            handoff_mask = handoff_cpu.to(self.device)
            canonical_ids = env_ids[~handoff_mask]
            if len(canonical_ids) > 0:
                base_payload = {
                    "scene_state": self._snapshot_state_single,
                    "initial_phase": self._snapshot_initial_phase,
                }
                self._restore_payload(
                    canonical_ids,
                    base_payload,
                    xy_offsets=self._sample_xy_offsets(len(canonical_ids)),
                    reset_mode=NORMAL,
                    handoff_source=False,
                )
            handoff_ids = env_ids[handoff_mask]
            if len(handoff_ids) > 0:
                picks = torch.randint(
                    0, len(self._handoff_snapshot_payloads), (len(handoff_ids),), generator=self._rng
                )
                for snap_idx in torch.unique(picks).tolist():
                    local = picks == int(snap_idx)
                    ids = handoff_ids[local.to(self.device)]
                    payload = self._handoff_snapshot_payloads[int(snap_idx)]
                    offsets = sample_uniform_xy(
                        len(ids), self.handoff_xy_jitter_m, self._rng
                    ).to(self.device)
                    self._restore_payload(
                        ids, payload, xy_offsets=offsets, reset_mode=NORMAL, handoff_source=True
                    )
            return

        if not randomize_target or self._curriculum_cfg.hard_fraction <= 0:
            offsets = self._sample_xy_offsets(len(env_ids)) if randomize_target else None
            base_payload = {
                "scene_state": self._snapshot_state_single,
                "initial_phase": self._snapshot_initial_phase,
            }
            self._restore_payload(env_ids, base_payload, xy_offsets=offsets, reset_mode=NORMAL)
            return

        modes_cpu = assign_reset_modes(len(env_ids), self._rng, self._curriculum_cfg)
        modes = modes_cpu.to(self.device)
        for mode in (NORMAL, EDGE):
            mask = modes == mode
            ids = env_ids[mask]
            if len(ids) == 0:
                continue
            offsets = self._sample_xy_offsets(len(ids)) if mode == NORMAL else self._sample_edge_offsets(len(ids))
            base_payload = {
                "scene_state": self._snapshot_state_single,
                "initial_phase": self._snapshot_initial_phase,
            }
            self._restore_payload(ids, base_payload, xy_offsets=offsets, reset_mode=mode)

        term_mask = modes == TERMINAL
        term_ids = env_ids[term_mask]
        if len(term_ids) > 0:
            # Random snapshot assignment on CPU for deterministic sampling.
            picks = torch.randint(
                0, len(self._terminal_snapshot_payloads), (len(term_ids),), generator=self._rng
            )
            for snap_idx in torch.unique(picks).tolist():
                local = picks == int(snap_idx)
                ids = term_ids[local.to(self.device)]
                payload = self._terminal_snapshot_payloads[int(snap_idx)]
                self._restore_payload(
                    ids, payload, xy_offsets=self._sample_terminal_offsets(len(ids)), reset_mode=TERMINAL
                )

    def curriculum_diagnostics(self) -> dict[str, float | int]:
        total = int(self._reset_mode_counts.sum().item())
        def frac(i: int) -> float:
            return float(self._reset_mode_counts[i].item() / total) if total else 0.0
        source_total = int(self._reset_source_counts.sum().item())
        canonical_frac = float(self._reset_source_counts[0].item() / source_total) if source_total else 0.0
        handoff_frac = float(self._reset_source_counts[1].item() / source_total) if source_total else 0.0
        return {
            "hard_case_fraction_requested": float(self._curriculum_cfg.hard_fraction),
            "handoff_fraction_requested": float(self.handoff_fraction),
            "handoff_snapshot_count": int(len(self._handoff_snapshot_payloads)),
            "handoff_xy_jitter_m": float(self.handoff_xy_jitter_m),
            "reset_source_canonical_frac": canonical_frac,
            "reset_source_handoff_frac": handoff_frac,
            "terminal_fraction_within_hard": float(self._curriculum_cfg.terminal_fraction_within_hard),
            "terminal_snapshot_count": int(len(self._terminal_snapshot_payloads)),
            "reset_normal_frac": frac(NORMAL),
            "reset_edge_frac": frac(EDGE),
            "reset_terminal_frac": frac(TERMINAL),
            "release_gate_steps": int(self._release_gate_cfg.gate_steps),
            "release_speed_threshold_mps": float(self._release_gate_cfg.speed_threshold_mps),
            "release_primitive_mode": str(self._release_primitive_cfg.mode),
            "release_open_hold_steps": int(self._release_primitive_cfg.open_hold_steps),
            "release_retreat_steps": int(self._release_primitive_cfg.retreat_steps),
            "release_retreat_dz_per_step": float(self._release_primitive_cfg.retreat_dz_per_step),
            "support_gate_steps": int(self._release_primitive_cfg.support_gate_steps),
            "support_force_threshold_n": float(self._release_primitive_cfg.support_force_threshold_n),
            "support_max_wait_steps": int(self._release_primitive_cfg.support_max_wait_steps),
            "support_backend": (
                "raw_physx_pair" if self._support_contact_provider is not None
                else ("filtered_sensor_legacy" if "red_blue_support_contact" in self._scene.sensors else "none")
            ),
            "learn_release_timing": bool(self.learn_release_timing),
            "release_signal_threshold": float(self._learned_release_cfg.threshold),
            "release_wait_free_steps": int(self._learned_release_cfg.wait_free_steps),
        }

    def restored_snapshot_diagnostics(self) -> dict[str, float]:
        """Return world/local diagnostics for smoke validation and logging."""
        m = self._read_measurements()
        origins = self._scene.env_origins
        red_local = m.red_pos - origins
        blue_local = m.blue_pos - origins
        xy = torch.linalg.vector_norm(m.red_pos[:, :2] - m.blue_pos[:, :2], dim=-1)
        geom_ready = (red_local[:, 2] > 0.05) & (xy < 0.15)
        return {
            "geom_ready_frac": float(geom_ready.float().mean().item()),
            "physical_grasp_frac": float(m.physical_grasp.float().mean().item()),
            "mean_xy_m": float(xy.mean().item()),
            "mean_red_local_z_m": float(red_local[:, 2].mean().item()),
            "env0_origin_x": float(origins[0, 0].item()),
            "env0_origin_y": float(origins[0, 1].item()),
            "env0_red_world_x": float(m.red_pos[0, 0].item()),
            "env0_red_world_y": float(m.red_pos[0, 1].item()),
            "env0_red_local_x": float(red_local[0, 0].item()),
            "env0_red_local_y": float(red_local[0, 1].item()),
            "env0_blue_local_x": float(blue_local[0, 0].item()),
            "env0_blue_local_y": float(blue_local[0, 1].item()),
        }

    # ------------------------------------------------------------------
    # State and observation
    # ------------------------------------------------------------------
    def _read_measurements(self) -> _Measurements:
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
        speed = torch.linalg.vector_norm(p.red_lin_vel_w, dim=-1)
        if self._support_contact_provider is not None:
            support_force = self._support_contact_provider.force_per_env(num_envs=self.num_envs)
            support_force = support_force.to(device=self.device, dtype=torch.float32)
        elif "red_blue_support_contact" in self._scene.sensors:
            # Legacy filtered-sensor fallback retained only for compatibility.
            from stage_vla.envs.state_readers import filtered_force_per_env
            support_force = filtered_force_per_env(self._scene.sensors["red_blue_support_contact"])
        else:
            support_force = torch.zeros_like(speed)
        return _Measurements(
            red_pos=p.red_pos_w,
            blue_pos=p.blue_pos_w,
            red_lin_vel=p.red_lin_vel_w,
            red_ang_vel=p.red_ang_vel_w,
            gripper_joint_pos=p.gripper_joint_pos,
            gripper_open=p.gripper_open,
            ee_pos=ee_pos,
            finger_forces=torch.stack([f_l, f_r], dim=-1),
            physical_grasp=physical,
            red_speed=speed,
            support_force_n=support_force,
        )

    def _build_bc_state(self, m: _Measurements, phase: torch.Tensor) -> torch.Tensor:
        offset = TARGET_OFFSET.to(self.device).reshape(1, 3)
        target = m.blue_pos + offset
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
            raise RuntimeError(f"BC state dim mismatch: {state.shape}")
        return state

    def _advance_state(self) -> tuple[torch.Tensor, torch.Tensor, _Measurements]:
        m = self._read_measurements()
        local_red_z = m.red_pos[:, 2] - self._scene.env_origins[:, 2]
        grace_grasp = (self._grasp_grace > 0) & (local_red_z > 0.05)
        physical_for_phase = m.physical_grasp | grace_grasp
        phase = self._phase_tracker.update(
            m.red_pos, m.blue_pos, m.red_speed, m.gripper_open, physical_for_phase
        )
        self._grasp_grace = torch.clamp(self._grasp_grace - 1, min=0)
        state = self._build_bc_state(m, phase)
        return state, phase, m

    def _state_after_direct_restore(self, env_ids: torch.Tensor) -> None:
        """Patch cache rows after snapshot restore without double phase updates."""
        if len(env_ids) == 0:
            return
        m = self._read_measurements()
        phase = self._cached_phase.clone()
        phase[env_ids] = self._phase_tracker.phase[env_ids]
        state = self._build_bc_state(m, phase)
        self._cached_state[env_ids] = state[env_ids]
        self._cached_phase[env_ids] = phase[env_ids]
        # Keep measurements coherent for logging on the next call.
        self._cached_measurements = m

    def _obs(self) -> TensorDict:
        if self._cached_state is None:
            raise RuntimeError("wrapper state not initialized")
        obs = residual_observation_from_bc_state(self._cached_state)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def get_observations(self):
        if self._cached_state is None:
            self._cached_state, self._cached_phase, self._cached_measurements = self._advance_state()
        return self._obs()

    def reset(self):
        self._restore_place(self._all_ids(), randomize_target=self.training_resets)
        self._cached_state, self._cached_phase, self._cached_measurements = self._advance_state()
        return self._obs(), {}

    # ------------------------------------------------------------------
    # Residual action, reward, termination
    # ------------------------------------------------------------------
    def _strict_success(self, m: _Measurements) -> torch.Tensor:
        xy = torch.linalg.vector_norm(m.red_pos[:, :2] - m.blue_pos[:, :2], dim=-1)
        z_err = torch.abs(m.red_pos[:, 2] - (m.blue_pos[:, 2] + TARGET_HEIGHT_DIFF_M))
        stacked = (xy < 0.04) & (z_err < 0.010)
        stable_now = stacked & m.gripper_open & (m.red_speed < 0.05)
        self._stable_count = torch.where(
            stable_now, self._stable_count + 1, torch.zeros_like(self._stable_count)
        )
        return self._stable_count >= self.stable_steps

    def step(self, actions):
        if self._cached_state is None:
            self.reset()
        policy_action = torch.as_tensor(actions, device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        prev_state = self._cached_state.clone()
        prev_phase = self._cached_phase.clone()
        if self.learn_release_timing:
            if policy_action.shape[-1] != LEARNED_RELEASE_ACTION_DIM:
                raise ValueError(f"M18 release actor must output 1 action, got {policy_action.shape[-1]}")
            release_signal = policy_action[:, 0]
            if self._frozen_movement_policy is None:
                raise RuntimeError("frozen A1 movement policy is missing")
            movement_obs = residual_observation_from_bc_state(prev_state)
            residual_unit = self._frozen_movement_policy(movement_obs).clamp(-1.0, 1.0)
        else:
            residual_unit = policy_action
            release_signal = torch.full((self.num_envs,), -1.0, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            base_cont, bc_grip = self.bc.act(prev_state)
        final_cont, residual_delta = compose_residual_cont_action(
            base_cont, residual_unit, prev_phase, self.residual_scales
        )
        # M17-A6 release primitive.  RELEASE motion is zero-locked by the
        # composer, then an explicit +Z retreat may be injected *only after*
        # the physical gripper is observed open.  R2 can gate OPEN on exact
        # filtered red<->blue support contact; it never uses object speed as a
        # hard release prerequisite.  BASELINE preserves prior M17/A5 behavior.
        prev_m = self._cached_measurements
        if prev_m is None:
            raise RuntimeError("cached measurements missing before action")
        learned_opened_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        learned_invalid_request = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        learned_waiting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.learn_release_timing:
            grip_cmd, learned_opened_now, learned_invalid_request, learned_waiting = self._learned_release.update(
                prev_phase, release_signal, self._episode_step
            )
            retreat_dz = torch.zeros(self.num_envs, device=self.device)
        elif self._release_primitive_cfg.mode == RELEASE_PRIMITIVE_BASELINE:
            prev_err = prev_state[:, 0:3]
            prev_xy = torch.linalg.vector_norm(prev_err[:, :2], dim=-1)
            prev_z = (prev_err[:, 2] - STRICT_Z_BIAS_M).abs()
            prev_speed = torch.linalg.vector_norm(prev_state[:, 6:9], dim=-1)
            release_open = self._release_gate.update(prev_phase, prev_xy, prev_z, prev_speed)
            if self._release_gate_cfg.gate_steps > 0:
                grip_cmd = torch.where(
                    release_open,
                    torch.ones_like(prev_phase, dtype=torch.float32),
                    -torch.ones_like(prev_phase, dtype=torch.float32),
                )
            else:
                grip_cmd = strict_grip_command(prev_phase)
            retreat_dz = torch.zeros(self.num_envs, device=self.device)
        else:
            grip_cmd, retreat_dz = self._release_primitive.update(
                prev_phase, prev_m.gripper_open, prev_m.support_force_n
            )
            final_cont = final_cont.clone()
            final_cont[:, 2] = final_cont[:, 2] + retreat_dz
        raw = torch.stack(
            [
                final_cont[:, 0],
                final_cont[:, 1],
                final_cont[:, 2],
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                final_cont[:, 3],
                grip_cmd,
            ],
            dim=-1,
        )

        # Physics backend only; base task rewards/terminations are ignored.
        _, _, _, _, base_extras = self.env.step(raw)
        self._prev_cont.copy_(final_cont.detach())
        self._episode_step += 1

        next_state, next_phase, next_m = self._advance_state()
        self._cached_state = next_state
        self._cached_phase = next_phase
        self._cached_measurements = next_m

        success = self._strict_success(next_m)
        fail = next_phase == FAIL
        timeout = self._episode_step >= self.place_episode_steps
        self.last_step_success.copy_(success)
        self.last_step_fail.copy_(fail)
        self.last_step_timeout.copy_(timeout)
        dones = success | fail | timeout

        reward = residual_place_reward(
            prev_state,
            next_state,
            prev_phase,
            next_phase,
            residual_unit,
            success,
            fail,
            cfg=self._reward_cfg,
        )
        if self.learn_release_timing:
            # Critical anti-exploit contract: once RELEASE is reached, keeping
            # the cube grasped must not farm the dense near-stack/settle reward.
            # Waiting gets zero reward for a tiny grace window, then a small
            # negative step cost. Invalid early OPEN requests get a tiny penalty.
            reward = apply_learned_release_reward_contract(
                reward,
                waiting=learned_waiting,
                release_wait=self._learned_release.release_wait,
                invalid_request=learned_invalid_request,
                cfg=self._learned_release_cfg,
            )
        self._episode_return += reward
        self._episode_success |= success

        extras = dict(base_extras) if isinstance(base_extras, dict) else {}
        extras.setdefault("log", {})
        red_err = next_state[:, 0:3]
        extras["log"].update(
            {
                "m17/xy_error_m": torch.linalg.vector_norm(red_err[:, :2], dim=-1).mean(),
                "m17/z_error_abs_m": red_err[:, 2].abs().mean(),
                "m17/red_speed_mps": next_m.red_speed.mean(),
                "m17/residual_abs_mean": residual_delta.abs().mean(),
                "m17/release_fraction": (next_phase == RELEASE).float().mean(),
                "m17/release_open_fraction": (
                    self._learned_release.opened.float().mean()
                    if self.learn_release_timing
                    else (
                        self._release_gate.opened.float().mean()
                        if self._release_primitive_cfg.mode == RELEASE_PRIMITIVE_BASELINE
                        else self._release_primitive.open_commanded.float().mean()
                    )
                ),
                "m18/release_signal_mean": release_signal.mean(),
                "m18/release_request_fraction": (release_signal > self._learned_release_cfg.threshold).float().mean(),
                "m18/release_opened_now_fraction": learned_opened_now.float().mean(),
                "m18/release_wait_mean": self._learned_release.release_wait.float().mean(),
                "m18/invalid_release_request_fraction": learned_invalid_request.float().mean(),
                "m17/release_gate_count_mean": self._release_gate.count.float().mean(),
                "m17/support_force_n_mean": next_m.support_force_n.mean(),
                "m17/support_gate_count_mean": self._release_primitive.support_count.float().mean(),
                "m17/retreat_active_fraction": (retreat_dz > 0).float().mean(),
                "m17/retreat_count_mean": self._release_primitive.retreat_count.float().mean(),
                "m17/support_fallback_fraction": self._release_primitive.fallback_open.float().mean(),
                "m17/success_step_fraction": success.float().mean(),
                "m17/bc_grip_disagree": ((bc_grip == 1) != (prev_phase == RELEASE)).float().mean(),
                "m17/reset_mode_normal": (self._last_reset_mode == NORMAL).float().mean(),
                "m17/reset_mode_edge": (self._last_reset_mode == EDGE).float().mean(),
                "m17/reset_mode_terminal": (self._last_reset_mode == TERMINAL).float().mean(),
                "m19/reset_source_handoff": self._last_reset_handoff.float().mean(),
            }
        )

        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if len(done_ids) > 0:
            # Log completed episodes *before* resetting their counters.
            extras["log"]["m17/episode_return_done"] = self._episode_return[done_ids].mean()
            extras["log"]["m17/episode_success_done"] = self._episode_success[done_ids].float().mean()
            extras["log"]["m17/episode_length_done"] = self._episode_step[done_ids].float().mean()
            if self.training_resets:
                self._restore_place(done_ids, randomize_target=True)
                self._state_after_direct_restore(done_ids)

        return self._obs(), reward, dones, extras
