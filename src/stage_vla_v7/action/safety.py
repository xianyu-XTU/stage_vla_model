"""Final normalized action projection shared by every policy backend."""

from __future__ import annotations

from dataclasses import dataclass

from stage_vla_v7.interfaces import RobotAction, Skill


HOLDING_SKILLS = frozenset(
    {Skill.GRASP, Skill.LIFT, Skill.TRANSPORT, Skill.ALIGN, Skill.DESCEND}
)
OPEN_SKILLS = frozenset({Skill.REACH, Skill.RELEASE_STABILIZE, Skill.RETREAT})


def project_skill_action(skill: Skill | str, action: object) -> object:
    """Clamp a batched 5D action and enforce the frozen gripper polarity."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - tensor bridge requires torch
        raise RuntimeError("batched action projection requires torch") from exc
    canonical = Skill(getattr(skill, "value", skill))
    value = torch.as_tensor(action)
    if value.ndim != 2 or value.shape[-1] != 5 or not torch.isfinite(value).all():
        raise ValueError("skill action must be finite with shape [N,5]")
    projected = value.clamp(-1.0, 1.0).clone()
    if canonical in HOLDING_SKILLS:
        projected[:, 4] = -projected[:, 4].abs()
    elif canonical in OPEN_SKILLS:
        projected[:, 4] = projected[:, 4].abs()
    else:  # pragma: no cover - Skill exhaustiveness guard
        raise ValueError(f"unsupported skill: {canonical.value!r}")
    return projected


def hold_finished_skill_action(
    skill: Skill | str,
    action: object,
    finished: object,
    last_action: object,
) -> object:
    """Hold pose and preserve grip at a vectorized Skill barrier."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - tensor bridge requires torch
        raise RuntimeError("batched action projection requires torch") from exc
    canonical = Skill(getattr(skill, "value", skill))
    value = torch.as_tensor(action)
    done = torch.as_tensor(finished, device=value.device)
    previous = torch.as_tensor(last_action, device=value.device, dtype=value.dtype)
    if value.ndim != 2 or value.shape[-1] != 5:
        raise ValueError("skill action must have shape [N,5]")
    if done.dtype != torch.bool or done.shape != (value.shape[0],):
        raise ValueError("finished must be a boolean vector matching the action batch")
    if previous.shape != value.shape:
        raise ValueError("last_action must match action shape")
    if not torch.isfinite(value).all() or not torch.isfinite(previous).all():
        raise ValueError("skill actions must be finite")
    held = value.clone()
    held[done, :4] = 0.0
    held[done, 4] = previous[done, 4]
    return project_skill_action(canonical, held)


def entrance_action_scale(
    age_steps: object,
    *,
    warmup_steps: int,
    start_scale: float,
) -> object:
    """Linearly hand arm control from ``start_scale`` to full authority."""
    import torch

    age = torch.as_tensor(age_steps)
    if age.ndim != 1:
        raise ValueError("warmup age must have shape [N]")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 < float(start_scale) <= 1.0:
        raise ValueError("start_scale must lie in (0,1]")
    if warmup_steps <= 1:
        return torch.ones_like(age, dtype=torch.float32)
    progress = age.to(torch.float32).clamp(0, warmup_steps - 1)
    progress = progress / float(warmup_steps - 1)
    return float(start_scale) + (1.0 - float(start_scale)) * progress


def carrying_grasp_lost(
    between_fingertips: object,
    finger_a_contact: object,
    finger_b_contact: object,
) -> object:
    """Detect an unrecoverable carry failure without rejecting contact tilt."""
    import torch

    between = torch.as_tensor(between_fingertips)
    contact_a = torch.as_tensor(finger_a_contact, device=between.device)
    contact_b = torch.as_tensor(finger_b_contact, device=between.device)
    if (
        between.ndim != 1
        or contact_a.shape != between.shape
        or contact_b.shape != between.shape
    ):
        raise ValueError("carry grasp masks must share shape [N]")
    if (
        between.dtype != torch.bool
        or contact_a.dtype != torch.bool
        or contact_b.dtype != torch.bool
    ):
        raise ValueError("carry grasp masks must be boolean")
    return ~between | ~contact_a | ~contact_b


def jaw_leveling_axis_angle(
    left_tip_w: object,
    right_tip_w: object,
    *,
    max_angle_rad: float = 0.02,
) -> object:
    """Return bounded world-frame roll/pitch that levels the jaw axis."""
    import torch

    left = torch.as_tensor(left_tip_w, dtype=torch.float32)
    right = torch.as_tensor(right_tip_w, device=left.device, dtype=left.dtype)
    if left.ndim != 2 or left.shape[-1] != 3 or right.shape != left.shape:
        raise ValueError("fingertip positions must share shape [N,3]")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("fingertip positions must be finite")
    if not 0 < float(max_angle_rad) <= torch.pi / 2:
        raise ValueError("max_angle_rad must lie in (0,pi/2]")
    jaw = right - left
    horizontal = jaw[:, :2].norm(dim=-1)
    if torch.any(horizontal < 1.0e-6):
        raise ValueError("parallel-jaw axis must have a horizontal component")
    tilt = torch.atan2(jaw[:, 2], horizontal).clamp(
        -float(max_angle_rad), float(max_angle_rad)
    )
    axis_xy = torch.stack((-jaw[:, 1], jaw[:, 0]), dim=-1) / horizontal.unsqueeze(-1)
    return axis_xy * tilt.unsqueeze(-1)


def project_pregrasp_edge_alignment(
    action: object,
    yaw_error_rad: object,
    contact_free: object,
    *,
    yaw_limit_rad: float = 0.02,
    tolerance_rad: float = 0.04,
) -> tuple[object, object]:
    """Keep an open, stationary pregrasp while the wrist aligns to an edge."""
    import torch

    value = torch.as_tensor(action)
    error = torch.as_tensor(yaw_error_rad, device=value.device, dtype=value.dtype)
    free = torch.as_tensor(contact_free, device=value.device, dtype=torch.bool)
    if value.ndim != 2 or value.shape[-1] != 5:
        raise ValueError("pregrasp action must have shape [N,5]")
    if error.shape != (len(value),) or free.shape != error.shape:
        raise ValueError("yaw error and contact mask must have shape [N]")
    if (
        not torch.isfinite(value).all()
        or not torch.isfinite(error).all()
        or not 0.0 < float(yaw_limit_rad) <= torch.pi / 2.0
        or not 0.0 <= float(tolerance_rad) < torch.pi / 4.0
    ):
        raise ValueError("pregrasp alignment inputs must be finite and bounded")
    aligning = free & (error.abs() > float(tolerance_rad))
    projected = value.clone()
    projected[free, 3] = 0.0
    projected[aligning, :3] = 0.0
    projected[aligning, 3] = (
        error[aligning] / float(yaw_limit_rad)
    ).clamp(-1.0, 1.0)
    projected[aligning, 4] = 1.0
    return projected, aligning


def object_upright_tilt_rad(quat_xyzw: object) -> object:
    """Return the angle between an object's local +Z axis and world +Z."""
    import torch

    quat = torch.as_tensor(quat_xyzw, dtype=torch.float32)
    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError("object quaternion must have shape [N,4] in XYZW order")
    if not torch.isfinite(quat).all():
        raise ValueError("object quaternion must be finite")
    norm = quat.norm(dim=-1, keepdim=True)
    if torch.any(norm < 1.0e-6):
        raise ValueError("object quaternion must be nonzero")
    qx, qy, _qz, _qw = (quat / norm).unbind(dim=-1)
    top_z = (1.0 - 2.0 * (qx.square() + qy.square())).clamp(-1.0, 1.0)
    return torch.acos(top_z)


def project_descend_motion(
    action: object,
    relative_xyz_m: object,
    target_height_m: object,
    *,
    xy_tolerance_m: float,
    inner_xy_m: float,
    translation_limit_m: float,
) -> object:
    """Bound DESCEND motion toward the planar and vertical target region."""
    import torch

    value = torch.as_tensor(action)
    relative = torch.as_tensor(relative_xyz_m, device=value.device, dtype=value.dtype)
    target = torch.as_tensor(target_height_m, device=value.device, dtype=value.dtype)
    if value.ndim != 2 or value.shape[-1] != 5:
        raise ValueError("DESCEND action must have shape [N,5]")
    if relative.shape != (value.shape[0], 3):
        raise ValueError("relative_xyz_m must have shape [N,3]")
    if target.shape != (value.shape[0],):
        raise ValueError("target_height_m must have shape [N]")
    if (
        not torch.isfinite(value).all()
        or not torch.isfinite(relative).all()
        or not torch.isfinite(target).all()
        or not torch.isfinite(
            torch.tensor([xy_tolerance_m, inner_xy_m, translation_limit_m])
        ).all()
        or not 0.0 < float(inner_xy_m) < float(xy_tolerance_m)
        or float(translation_limit_m) <= 0.0
    ):
        raise ValueError("DESCEND projection inputs must be finite and valid")
    projected = value.clone()
    xy = relative[:, :2]
    distance = xy.norm(dim=-1)
    inward = -xy / distance.clamp_min(1.0e-9).unsqueeze(-1)
    inward_command = (value[:, :2] * inward).sum(dim=-1).clamp_min(0.0)
    max_command = (
        (distance - float(inner_xy_m)).clamp_min(0.0)
        / float(translation_limit_m)
    ).clamp_max(1.0)
    projected[:, :2] = inward * torch.minimum(
        inward_command, max_command
    ).unsqueeze(-1)
    aligned = distance <= float(xy_tolerance_m)
    above_target = relative[:, 2] > target
    projected[:, 2] = torch.where(
        aligned & above_target,
        projected[:, 2].clamp(max=0.0),
        torch.zeros_like(projected[:, 2]),
    )
    return projected


def project_align_motion(
    action: object,
    relative_xyz_m: object,
    target_height_m: object,
    *,
    translation_limit_m: float,
    planar_height_margin_m: float | None = None,
) -> object:
    """Keep ALIGN translation directed toward the object/support target."""
    import torch

    value = torch.as_tensor(action)
    relative = torch.as_tensor(relative_xyz_m, device=value.device, dtype=value.dtype)
    target = torch.as_tensor(target_height_m, device=value.device, dtype=value.dtype)
    limit = float(translation_limit_m)
    height_margin = (
        None if planar_height_margin_m is None else float(planar_height_margin_m)
    )
    if value.ndim != 2 or value.shape[-1] != 5:
        raise ValueError("ALIGN action must have shape [N,5]")
    if relative.shape != (value.shape[0], 3):
        raise ValueError("relative_xyz_m must have shape [N,3]")
    if target.shape != (value.shape[0],):
        raise ValueError("target_height_m must have shape [N]")
    if (
        not torch.isfinite(value).all()
        or not torch.isfinite(relative).all()
        or not torch.isfinite(target).all()
        or not torch.isfinite(torch.tensor(limit))
        or limit <= 0.0
        or (
            height_margin is not None
            and (
                not torch.isfinite(torch.tensor(height_margin))
                or not 0.0 <= height_margin <= limit
            )
        )
    ):
        raise ValueError("ALIGN projection inputs must be finite and valid")
    error = relative.clone()
    error[:, 2] -= target
    desired = -error / limit
    moving_toward_target = value[:, :3] * desired > 0.0
    bounded_magnitude = torch.minimum(value[:, :3].abs(), desired.abs())
    projected = value.clone()
    projected[:, :3] = torch.where(
        moving_toward_target,
        desired.sign() * bounded_magnitude,
        torch.zeros_like(bounded_magnitude),
    )
    if height_margin is not None:
        planar_ready = relative[:, 2] >= target - height_margin
        projected[~planar_ready, :2] = 0.0
        projected[~planar_ready, 3] = 0.0
    return projected


@dataclass(frozen=True)
class SafetyLimits:
    """Normalized command limits applied after model inference."""

    translation: float = 1.0
    yaw: float = 1.0
    grip_min: float = -1.0
    grip_max: float = 1.0

    def __post_init__(self) -> None:
        if self.translation <= 0.0 or self.yaw <= 0.0 or self.grip_min > self.grip_max:
            raise ValueError("invalid action safety limits")


class SafetyProjector:
    """Clamp model output and hold finished skills in a safe state."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    @staticmethod
    def _finished_grip(skill: Skill) -> float:
        return 1.0 if skill in {Skill.REACH, Skill.RELEASE_STABILIZE, Skill.RETREAT} else -1.0

    def project(self, skill: Skill, action: RobotAction, *, finished: bool = False) -> RobotAction:
        limits = self.limits
        if finished:
            grip = min(limits.grip_max, max(limits.grip_min, self._finished_grip(skill)))
            return RobotAction(0.0, 0.0, 0.0, 0.0, grip)
        dx, dy, dz, dyaw, grip = action.values
        return RobotAction(
            min(limits.translation, max(-limits.translation, dx)),
            min(limits.translation, max(-limits.translation, dy)),
            min(limits.translation, max(-limits.translation, dz)),
            min(limits.yaw, max(-limits.yaw, dyaw)),
            min(limits.grip_max, max(limits.grip_min, grip)),
        )
