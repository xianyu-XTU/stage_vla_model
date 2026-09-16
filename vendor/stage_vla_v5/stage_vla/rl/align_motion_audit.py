"""M29 read-only motion diagnostics. XYZW world rotations; no scoring changes."""
import math
import torch


def normalized_quat(q):
    q = torch.as_tensor(q, dtype=torch.float64)
    if q.shape[-1] != 4 or not torch.isfinite(q).all() or (q.norm(dim=-1) < 1e-12).any():
        raise ValueError("finite nonzero XYZW quaternions required")
    return q / q.norm(dim=-1, keepdim=True)


def quat_mul(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat([aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)], -1)


def rotation_vector_world(before, after):
    """Shortest world-frame rotation from before to after, radians.

    Sign-invariant, stable at identity. Cannot resolve >pi rotation within a
    sampling interval; this is a diagnostic finite difference, not truth at an
    unobserved solver iteration and never a substitute for the task velocity.
    """
    a, b = normalized_quat(before), normalized_quat(after)
    inverse = torch.cat([-a[..., :3], a[..., 3:]], -1)
    delta = quat_mul(b, inverse)
    delta = torch.where(delta[..., 3:] < 0, -delta, delta)
    size = delta[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(size, delta[..., 3:].clamp_min(0))
    scale = torch.where(size > 1e-12, angle / size.clamp_min(1e-12), torch.full_like(size, 2.))
    return delta[..., :3] * scale


def angular_velocity_fd(before, after, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite sample dt required")
    return rotation_vector_world(before, after) / dt


def predicted_quat(before, omega, dt):
    q = normalized_quat(before)
    omega = torch.as_tensor(omega, dtype=q.dtype, device=q.device)
    angle = omega.norm(dim=-1, keepdim=True) * dt
    axis = omega / omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    delta = torch.cat([axis * torch.sin(angle / 2), torch.cos(angle / 2)], -1)
    return quat_mul(delta, q)


def quantiles(values):
    x = torch.as_tensor(values).double().flatten()
    if not x.numel():
        return {"n": 0, "median": None, "p95": None, "max": None}
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite audit values")
    return {"n": x.numel(), "median": float(x.median()), "p95": float(torch.quantile(x, .95)), "max": float(x.max())}


def analyze_trace(trace, *, burn_in_s=.5):
    """Summarize all sources without removing failed or already-ended entries.

    Ended rows remain in metadata but their later physical samples are masked.
    Probe mode explicitly records every frame, without policy success claims.
    """
    f = trace["frames"]
    dt = trace["sample_dt_s"]
    t, n = f["pos"].shape[:2]
    if t < 2 or len(trace["sources"]) != n:
        raise ValueError("missing samples or source identities")
    if f["active"].shape != (t, n):
        raise ValueError("invalid active mask")
    omega_fd = angular_velocity_fd(f["quat_xyzw"][:-1], f["quat_xyzw"][1:], dt)
    omega = f["omega"][1:].double()
    lin_fd = (f["com_pos"][1:].double() - f["com_pos"][:-1].double()) / dt
    active = f["active"][1:].bool()
    elapsed = torch.arange(1, t, dtype=torch.float64)[:, None] * dt
    steady = active & (elapsed >= burn_in_s)
    held = f["held"][1:].bool()
    # Compare local alias to direct PhysX retrieval at exactly the same instant.
    angular_api_error = (f["omega"] - f["raw_velocity"][..., 3:]).norm(dim=-1)
    linear_api_error = (f["lin"] - f["raw_velocity"][..., :3]).norm(dim=-1)
    position_api_error = (f["pos"] - f["raw_pose"][..., :3]).norm(dim=-1)
    quaternion_api_error = rotation_vector_world(f["quat_xyzw"], f["raw_pose"][..., 3:]).norm(dim=-1)
    avg_omega = (f["omega"][:-1].double() + omega) / 2
    predicted = predicted_quat(f["quat_xyzw"][:-1], avg_omega, dt)
    residual_angle = rotation_vector_world(predicted, f["quat_xyzw"][1:]).norm(dim=-1)
    result = {"samples_per_env": t, "sample_dt_s": dt, "burn_in_s": burn_in_s,
        "angular_api_max_error_radps": float(angular_api_error.max()),
        "linear_api_max_error_mps": float(linear_api_error.max()),
        "position_api_max_error_m": float(position_api_error.max()),
        "quaternion_api_max_error_rad": float(quaternion_api_error.max()),
        "scope": "motion audit, finite differences are diagnostics only; no changed success rate", "per_env": []}
    for i, source in enumerate(trace["sources"]):
        mask = steady[:, i]
        held_mask = mask & held[:, i]
        w, fd = omega[:, i].norm(dim=-1), omega_fd[:, i].norm(dim=-1)
        row = {"env": i, "source_seed": source, "scored_samples": int(active[:, i].sum()),
            "steady_samples": int(mask.sum()), "steady_held_samples": int(held_mask.sum()),
            "omega_api_radps": quantiles(w[mask]), "omega_pose_fd_radps": quantiles(fd[mask]),
            "omega_vector_error_radps": quantiles((omega[:, i] - omega_fd[:, i]).norm(dim=-1)[mask]),
            "linear_api_mps": quantiles(f["lin"][1:, i].norm(dim=-1)[mask]),
            "linear_com_fd_mps": quantiles(lin_fd[:, i].norm(dim=-1)[mask]),
            "quaternion_prediction_residual_rad": quantiles(residual_angle[:, i][mask]),
            "pose_rotation_path_rad": float((omega_fd[:, i].norm(dim=-1) * dt)[mask].sum()),
            "api_integrated_angular_speed_rad": float((w * dt)[mask].sum()),
            "held_api_over1_pose_fd_under1": int((held_mask & (w >= 1) & (fd < 1)).sum()),
            "both_over1": int((held_mask & (w >= 1) & (fd >= 1)).sum()),
            "held_force_n": quantiles(f["force"][1:, i][held_mask]),
            "max_step_grip_change_m": float((f["grip"][1:, i] - f["grip"][:-1, i]).abs().max()),
            "max_step_red_relative_ee_change_m": float(((f["pos"][1:, i] - f["ee"][1:, i]) - (f["pos"][:-1, i] - f["ee"][:-1, i])).norm(dim=-1)[active[:, i]].max())}
        result["per_env"].append(row)
    result["overall"] = {"steady_samples": int(steady.sum()), "steady_held_samples": int((steady & held).sum()),
        "omega_api_radps": quantiles(omega.norm(dim=-1)[steady]),
        "omega_pose_fd_radps": quantiles(omega_fd.norm(dim=-1)[steady]),
        "held_api_over1_pose_fd_under1": sum(r["held_api_over1_pose_fd_under1"] for r in result["per_env"]),
        "both_over1": sum(r["both_over1"] for r in result["per_env"])}
    return result
