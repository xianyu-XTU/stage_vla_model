"""M19-D2 frozen-A1 handoff recovery primitives.

M19-D1 showed that fine-tuning A1 on live TRANSPORT->PLACE handoffs caused
catastrophic forgetting.  M19-D2 therefore freezes A1_RF150 permanently and
trains a separate small adapter.  The adapter is active only before PLACE and
its sole objective is to move/damp a live handoff into the entry distribution
that the frozen A1 controller already handles.

This module is Isaac-independent so the transition contract, reward and actor
loading can be regression-tested with ordinary Python/PyTorch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

RECOVERY_ACTION_DIM = 4

GEOMETRY_LIFT = 0
GEOMETRY_ALIGN = 1
GEOMETRY_DESCEND = 2
GEOMETRY_HOLD = 3


@dataclass(frozen=True)
class RecoveryReadyConfig:
    """TRAIN-derived envelope for one-way RECOVERY -> A1 handoff.

    ``strict_z_center_m`` is the canonical A1 snapshot's red center height
    error relative to ``blue_z + 0.0468``.  We do not force the cube to the
    final stack height during recovery; we reproduce the *entry* height A1 was
    trained from.
    """

    xy_max_m: float
    strict_z_center_m: float
    strict_z_tolerance_m: float
    red_speed_max_mps: float
    consecutive_steps: int = 3

    def validate(self) -> None:
        if self.xy_max_m <= 0:
            raise ValueError("xy_max_m must be > 0")
        if self.strict_z_tolerance_m <= 0:
            raise ValueError("strict_z_tolerance_m must be > 0")
        if self.red_speed_max_mps <= 0:
            raise ValueError("red_speed_max_mps must be > 0")
        if self.consecutive_steps <= 0:
            raise ValueError("consecutive_steps must be > 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GeometryRecoveryConfig:
    """Deterministic TRANSPORT->PLACE stabilizer using simulator geometry.

    The accepted expert prefix ends very close to the blue cube.  When the red
    cube still has appreciable velocity, immediately asking A1 to descend keeps
    the grasped cube in contact with the support and can sustain an oscillation.
    This controller first creates vertical clearance, aligns in free space,
    then returns to the canonical A1 entry height and holds until the normal
    recovery-ready latch fires.

    Commands are raw Isaac IK-relative actions (the task applies its own action
    scale after this controller).
    """

    clearance_z_error_m: float = 0.035
    align_xy_tolerance_m: float = 0.012
    realign_xy_threshold_m: float = 0.040
    position_gain_raw_per_m: float = 2.0
    max_xy_raw: float = 0.006
    max_z_raw: float = 0.006
    z_tolerance_m: float = 0.003

    def validate(self) -> None:
        if self.clearance_z_error_m <= 0:
            raise ValueError("clearance_z_error_m must be > 0")
        if self.align_xy_tolerance_m <= 0:
            raise ValueError("align_xy_tolerance_m must be > 0")
        if self.realign_xy_threshold_m < self.align_xy_tolerance_m:
            raise ValueError("realign_xy_threshold_m must be >= align_xy_tolerance_m")
        if self.position_gain_raw_per_m <= 0:
            raise ValueError("position_gain_raw_per_m must be > 0")
        if self.max_xy_raw <= 0 or self.max_z_raw <= 0:
            raise ValueError("max_xy_raw and max_z_raw must be > 0")
        if self.z_tolerance_m <= 0:
            raise ValueError("z_tolerance_m must be > 0")


class GeometryRecoveryController:
    """Vectorized four-phase geometry recovery planner.

    This class is physics-independent.  Call :meth:`update` once per control
    step and execute the returned ``[dx, dy, dz, dyaw]`` raw action while
    keeping the gripper closed.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        ready_cfg: RecoveryReadyConfig,
        cfg: GeometryRecoveryConfig | None = None,
    ):
        self.device = torch.device(device)
        self.ready_cfg = ready_cfg
        self.ready_cfg.validate()
        self.cfg = cfg or GeometryRecoveryConfig()
        self.cfg.validate()
        self.phase = torch.full((num_envs,), GEOMETRY_LIFT, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.phase.fill_(GEOMETRY_LIFT)
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.phase[ids] = GEOMETRY_LIFT

    def update(
        self,
        *,
        red_pos_w: torch.Tensor,
        blue_pos_w: torch.Tensor,
        strict_z_error_m: torch.Tensor,
    ) -> torch.Tensor:
        red = red_pos_w.to(self.device, dtype=torch.float32)
        blue = blue_pos_w.to(self.device, dtype=torch.float32)
        z_error = strict_z_error_m.to(self.device, dtype=torch.float32)
        if red.shape != blue.shape or red.ndim != 2 or red.shape[-1] != 3:
            raise ValueError("red_pos_w and blue_pos_w must be matching [N,3] tensors")
        if z_error.shape != red.shape[:-1]:
            raise ValueError("strict_z_error_m must be [N]")

        xy_delta = blue[:, :2] - red[:, :2]
        xy_error = torch.linalg.vector_norm(xy_delta, dim=-1)

        lift_done = z_error >= float(self.cfg.clearance_z_error_m - self.cfg.z_tolerance_m)
        self.phase = torch.where(
            (self.phase == GEOMETRY_LIFT) & lift_done,
            torch.full_like(self.phase, GEOMETRY_ALIGN),
            self.phase,
        )
        align_done = xy_error <= float(self.cfg.align_xy_tolerance_m)
        self.phase = torch.where(
            (self.phase == GEOMETRY_ALIGN) & align_done,
            torch.full_like(self.phase, GEOMETRY_DESCEND),
            self.phase,
        )
        need_realign = (self.phase >= GEOMETRY_DESCEND) & (
            xy_error > float(self.cfg.realign_xy_threshold_m)
        )
        self.phase = torch.where(need_realign, torch.full_like(self.phase, GEOMETRY_LIFT), self.phase)

        target_z = float(self.ready_cfg.strict_z_center_m)
        descend_done = (z_error - target_z).abs() <= float(self.cfg.z_tolerance_m)
        self.phase = torch.where(
            (self.phase == GEOMETRY_DESCEND) & descend_done,
            torch.full_like(self.phase, GEOMETRY_HOLD),
            self.phase,
        )

        action = torch.zeros((red.shape[0], RECOVERY_ACTION_DIM), dtype=torch.float32, device=self.device)
        lift = self.phase == GEOMETRY_LIFT
        align = self.phase == GEOMETRY_ALIGN
        descend = self.phase == GEOMETRY_DESCEND

        lift_z_error = float(self.cfg.clearance_z_error_m) - z_error
        action[:, 2] = torch.where(
            lift,
            (lift_z_error * float(self.cfg.position_gain_raw_per_m)).clamp(
                -float(self.cfg.max_z_raw), float(self.cfg.max_z_raw)
            ),
            action[:, 2],
        )
        xy_cmd = (xy_delta * float(self.cfg.position_gain_raw_per_m)).clamp(
            -float(self.cfg.max_xy_raw), float(self.cfg.max_xy_raw)
        )
        action[:, :2] = torch.where(align.unsqueeze(-1), xy_cmd, action[:, :2])
        descend_z_error = target_z - z_error
        action[:, 2] = torch.where(
            descend,
            (descend_z_error * float(self.cfg.position_gain_raw_per_m)).clamp(
                -float(self.cfg.max_z_raw), float(self.cfg.max_z_raw)
            ),
            action[:, 2],
        )
        return action


@dataclass(frozen=True)
class RecoveryRewardConfig:
    xy_progress_weight: float = 30.0
    speed_progress_weight: float = 3.0
    z_progress_weight: float = 15.0
    near_xy_weight: float = 0.15
    low_speed_weight: float = 0.20
    action_l2_weight: float = 0.01
    ready_bonus: float = 8.0
    lost_grasp_penalty: float = 8.0
    timeout_penalty: float = 1.0


def derive_ready_config(
    canonical_metrics: Mapping,
    *,
    a1_target_xy_range_m: float,
    xy_margin_m: float = 0.003,
    speed_multiplier: float = 1.5,
    speed_floor_mps: float = 0.10,
    strict_z_tolerance_m: float = 0.015,
    consecutive_steps: int = 3,
) -> RecoveryReadyConfig:
    """Derive the recovery target from TRAIN/canonical evidence only.

    A1 was trained by independently randomizing target X/Y within
    ``+/- a1_target_xy_range_m``.  The corresponding radial support is therefore
    ``sqrt(2) * range``.  The dynamic speed threshold is anchored to the
    canonical snapshot, with a small floor so a very quiet snapshot does not
    create an unrealistically strict gate.
    """
    if a1_target_xy_range_m <= 0:
        raise ValueError("a1_target_xy_range_m must be > 0")
    canonical_xy = float(canonical_metrics["red_blue_xy_m"])
    canonical_speed = float(canonical_metrics["red_speed_mps"])
    canonical_z = float(canonical_metrics["strict_z_error_m"])
    xy_max = canonical_xy + sqrt(2.0) * float(a1_target_xy_range_m) + float(xy_margin_m)
    speed_max = max(float(speed_floor_mps), canonical_speed * float(speed_multiplier))
    cfg = RecoveryReadyConfig(
        xy_max_m=xy_max,
        strict_z_center_m=canonical_z,
        strict_z_tolerance_m=float(strict_z_tolerance_m),
        red_speed_max_mps=speed_max,
        consecutive_steps=int(consecutive_steps),
    )
    cfg.validate()
    return cfg


def ready_now(
    *,
    xy_error_m: torch.Tensor,
    strict_z_error_m: torch.Tensor,
    red_speed_mps: torch.Tensor,
    physical_grasp: torch.Tensor,
    gripper_open: torch.Tensor,
    cfg: RecoveryReadyConfig,
) -> torch.Tensor:
    cfg.validate()
    return (
        (xy_error_m <= float(cfg.xy_max_m))
        & ((strict_z_error_m - float(cfg.strict_z_center_m)).abs() <= float(cfg.strict_z_tolerance_m))
        & (red_speed_mps <= float(cfg.red_speed_max_mps))
        & physical_grasp.bool()
        & ~gripper_open.bool()
    )


class RecoveryReadyTracker:
    """Consecutive-frame confirmation with a one-way latch."""

    def __init__(self, num_envs: int, device: str | torch.device, cfg: RecoveryReadyConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.latched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.count.zero_()
            self.latched.zero_()
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.count[ids] = 0
        self.latched[ids] = False

    def update(self, now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        now = now.to(self.device, dtype=torch.bool)
        active = ~self.latched
        self.count = torch.where(active & now, self.count + 1, torch.where(active, torch.zeros_like(self.count), self.count))
        newly = active & (self.count >= int(self.cfg.consecutive_steps))
        self.latched |= newly
        return self.latched.clone(), newly


def compose_recovery_action(
    frozen_a1_cont: torch.Tensor,
    adapter_unit: torch.Tensor,
    adapter_scale: torch.Tensor,
    *,
    max_abs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a bounded adapter correction without changing frozen A1 weights."""
    if frozen_a1_cont.shape != adapter_unit.shape or frozen_a1_cont.shape[-1] != RECOVERY_ACTION_DIM:
        raise ValueError("frozen_a1_cont and adapter_unit must be matching [...,4]")
    scale = torch.as_tensor(adapter_scale, dtype=frozen_a1_cont.dtype, device=frozen_a1_cont.device)
    if scale.shape != (RECOVERY_ACTION_DIM,):
        raise ValueError("adapter_scale must be [4]")
    delta = adapter_unit.clamp(-1.0, 1.0) * scale
    final = frozen_a1_cont + delta
    if max_abs is not None:
        lim = torch.as_tensor(max_abs, dtype=final.dtype, device=final.device)
        if lim.shape != (RECOVERY_ACTION_DIM,):
            raise ValueError("max_abs must be [4]")
        final = torch.maximum(torch.minimum(final, lim), -lim)
    return final, delta


def recovery_reward(
    *,
    prev_xy: torch.Tensor,
    next_xy: torch.Tensor,
    prev_speed: torch.Tensor,
    next_speed: torch.Tensor,
    prev_z_abs: torch.Tensor,
    next_z_abs: torch.Tensor,
    adapter_unit: torch.Tensor,
    ready: torch.Tensor,
    lost_grasp: torch.Tensor,
    timeout: torch.Tensor,
    ready_cfg: RecoveryReadyConfig,
    reward_cfg: RecoveryRewardConfig | None = None,
) -> torch.Tensor:
    cfg = reward_cfg or RecoveryRewardConfig()
    rew = cfg.xy_progress_weight * (prev_xy - next_xy)
    rew = rew + cfg.speed_progress_weight * (prev_speed - next_speed)
    rew = rew + cfg.z_progress_weight * (prev_z_abs - next_z_abs)
    rew = rew + cfg.near_xy_weight * torch.exp(-next_xy / max(float(ready_cfg.xy_max_m), 1e-6))
    rew = rew + cfg.low_speed_weight * torch.exp(-next_speed / max(float(ready_cfg.red_speed_max_mps), 1e-6))
    rew = rew - cfg.action_l2_weight * adapter_unit.pow(2).sum(dim=-1)
    rew = rew + cfg.ready_bonus * ready.float()
    rew = rew - cfg.lost_grasp_penalty * lost_grasp.float()
    rew = rew - cfg.timeout_penalty * timeout.float()
    return rew


class FrozenRecoveryAdapterPolicy(nn.Module):
    """Deterministic mean of an RSL-RL Gaussian adapter checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device):
        super().__init__()
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        actor = ck.get("actor_state_dict")
        if not isinstance(actor, dict):
            raise KeyError("recovery checkpoint missing actor_state_dict")
        weight_keys = sorted(
            [k for k, v in actor.items() if k.startswith("mlp.") and k.endswith(".weight") and torch.is_tensor(v)],
            key=lambda k: int(k.split(".")[1]),
        )
        if not weight_keys:
            raise ValueError("recovery actor has no mlp.*.weight tensors")
        layers: list[nn.Module] = []
        for i, w_key in enumerate(weight_keys):
            w = actor[w_key]
            b = actor.get(w_key[:-6] + "bias")
            if not torch.is_tensor(b):
                raise KeyError(f"missing bias for {w_key}")
            linear = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                linear.weight.copy_(w)
                linear.bias.copy_(b)
            layers.append(linear)
            if i != len(weight_keys) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers).to(device).eval()
        self.net.requires_grad_(False)
        self.obs_dim = int(actor[weight_keys[0]].shape[1])
        self.action_dim = int(actor[weight_keys[-1]].shape[0])
        if self.action_dim != RECOVERY_ACTION_DIM:
            raise ValueError(f"recovery actor must output 4 actions, got {self.action_dim}")

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"recovery obs expected {self.obs_dim}, got {obs.shape[-1]}")
        return self.net(obs)


def sha256_file(path: str | Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
