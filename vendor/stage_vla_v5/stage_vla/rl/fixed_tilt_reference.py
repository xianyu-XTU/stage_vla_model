"""Pure-torch fixed roll/pitch reference used by the isolated IK probe."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy, sy = torch.cos(yaw * .5), torch.sin(yaw * .5)
    cr, sr = torch.cos(roll * .5), torch.sin(roll * .5)
    cp, sp = torch.cos(pitch * .5), torch.sin(pitch * .5)
    return torch.stack((cy * sr * cp - sy * cr * sp,
                        cy * cr * sp + sy * sr * cp,
                        sy * cr * cp - cy * sr * sp,
                        cy * cr * cp + sy * sr * sp), dim=-1)


def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if quat.ndim != 2 or quat.shape[1] != 4 or not torch.isfinite(quat).all():
        raise ValueError("quaternion must be finite [N,4] XYZW")
    qx, qy, qz, qw = quat.unbind(-1)
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx.square() + qy.square()))
    sin_pitch = (2 * (qw * qy - qz * qx)).clamp(-1, 1)
    pitch = torch.asin(sin_pitch)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))
    return roll, pitch, yaw


class FixedTiltReference:
    """Latch roll/pitch after reset while integrating policy-provided yaw."""

    def __init__(self, num_envs: int, device: str | torch.device):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.reference_rpy = torch.zeros(num_envs, 3, device=device)
        self.desired_yaw = torch.zeros(num_envs, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.initialized.fill_(False)
        else:
            self.initialized[env_ids] = False

    def target(self, current_quat: torch.Tensor, delta_yaw: torch.Tensor) -> torch.Tensor:
        if current_quat.shape != (len(self.initialized), 4) or delta_yaw.shape != self.desired_yaw.shape:
            raise ValueError("one current quaternion and yaw command per environment required")
        if not torch.isfinite(delta_yaw).all():
            raise ValueError("yaw command contains non-finite values")
        roll, pitch, yaw = euler_xyz_from_quat(current_quat)
        pending = ~self.initialized
        self.reference_rpy[pending] = torch.stack((roll[pending], pitch[pending], yaw[pending]), dim=-1)
        self.desired_yaw[pending] = yaw[pending]
        self.initialized[pending] = True
        self.desired_yaw[:] = wrap_angle(self.desired_yaw + delta_yaw)
        return quat_from_euler_xyz(self.reference_rpy[:, 0], self.reference_rpy[:, 1], self.desired_yaw)


def summarize_tilt_trace(trace: dict, *, max_tilt_error_rad: float = .035) -> dict:
    """Summarize actual and desired tilt errors without altering task scoring."""
    frames = trace.get("frames", {})
    required = ("ee_quat_xyzw", "ik_desired_quat_xyzw", "tilt_reference_rpy", "tilt_initialized")
    if not all(name in frames for name in required):
        raise ValueError("trace lacks fixed-tilt pose fields")
    actual = torch.as_tensor(frames["ee_quat_xyzw"])
    desired = torch.as_tensor(frames["ik_desired_quat_xyzw"])
    reference = torch.as_tensor(frames["tilt_reference_rpy"])
    initialized = torch.as_tensor(frames["tilt_initialized"], dtype=torch.bool)
    if actual.ndim != 3 or actual.shape[-1] != 4 or desired.shape != actual.shape:
        raise ValueError("pose trace must be [T,N,4]")
    ar, ap, _ = euler_xyz_from_quat(actual.reshape(-1, 4))
    dr, dp, _ = euler_xyz_from_quat(desired.reshape(-1, 4))
    ar, ap = ar.reshape(actual.shape[:2]), ap.reshape(actual.shape[:2])
    dr, dp = dr.reshape(actual.shape[:2]), dp.reshape(actual.shape[:2])
    actual_error = torch.maximum(wrap_angle(ar - reference[..., 0]).abs(),
                                 wrap_angle(ap - reference[..., 1]).abs())
    desired_error = torch.maximum(wrap_angle(dr - reference[..., 0]).abs(),
                                  wrap_angle(dp - reference[..., 1]).abs())
    rows = []
    for env in range(actual.shape[1]):
        mask = initialized[:, env]
        if not mask.any():
            raise ValueError("fixed-tilt reference was never initialized")
        actual_max = float(actual_error[mask, env].max())
        desired_max = float(desired_error[mask, env].max())
        rows.append({"env": env, "max_actual_tilt_error_rad": actual_max,
                     "max_actual_tilt_error_deg": actual_max * 180 / torch.pi,
                     "max_ik_target_tilt_error_rad": desired_max,
                     "passed": actual_max <= max_tilt_error_rad and desired_max <= 1e-5})
    return {"version": "M31-fixed-tilt-diagnostic-v1", "max_tilt_error_rad": max_tilt_error_rad,
            "rows": rows, "passed": all(row["passed"] for row in rows)}
