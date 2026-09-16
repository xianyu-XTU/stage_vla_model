"""M30 shadow grasp validation. Never used by a policy or task scorer."""
from dataclasses import asdict, dataclass
import math

import torch

from .stageppo_align_core import align_grasp
from .align_motion_audit import angular_velocity_fd, quantiles, rotation_vector_world


@dataclass(frozen=True)
class GripCalibrationContract:
    version: str = "M30-shadow-asymmetric-grasp-v1"
    one_finger_below_m: float = .038
    joint_min_m: float = -.001
    joint_max_m: float = .041
    positive_fraction: float = .98
    max_missing_controls: int = 1
    follow_error_m: float = .002
    lift_min_m: float = .003
    lateral_min_m: float = .001
    yaw_min_rad: float = .001
    empty_clearance_m: float = .060
    empty_finger_max_m: float = .010
    max_angular_radps: float = 1.
    max_linear_mps: float = .05


CONTRACT = GripCalibrationContract()
# Fixed open-loop calibration only. No target-dependent action or PPO data.
SEQUENCE = [
    ("closed_neutral", 40, [0, 0, 0, 0, -1]),
    ("small_lift", 20, [0, 0, .3, 0, -1]),
    ("small_yaw", 20, [0, 0, 0, .15, -1]),
    ("closed_neutral_after_yaw", 20, [0, 0, 0, 0, -1]),
    ("x_positive", 20, [.3, 0, 0, 0, -1]),
    ("x_negative", 20, [-.3, 0, 0, 0, -1]),
    ("y_positive", 20, [0, .3, 0, 0, -1]),
    ("y_negative", 20, [0, -.3, 0, 0, -1]),
    ("yaw_reverse", 20, [0, 0, 0, -.15, -1]),
    ("closed_settle", 40, [0, 0, 0, 0, -1]),
    ("open_release", 20, [0, 0, 0, 0, 1]),
    ("open_separate", 80, [0, 0, .8, 0, 1]),
    ("empty_close", 20, [0, 0, 0, 0, -1]),
    ("empty_hold", 20, [0, 0, 0, 0, -1]),
    ("empty_lift", 20, [0, 0, .3, 0, -1]),
    ("open_again", 20, [0, 0, 0, 0, 1]),
]


def shadow_grasp(m, c=CONTRACT):
    """Keep original M4 geometry/contact; allow asymmetric measured closure.

    Return old and shadow flags side by side. No actions or temporal state.
    Net force is not contact-pair identity; calibration needs motion negatives.
    """
    for key in ("red", "left_tip", "right_tip", "force", "grip"):
        if not torch.isfinite(m[key]).all():
            raise ValueError(f"nonfinite {key}")
    old, _, d = align_grasp(m)
    plausible = ((m["grip"] >= c.joint_min_m) & (m["grip"] <= c.joint_max_m)).all(-1)
    closure = (m["grip"] < c.one_finger_below_m).any(-1)
    shadow = d.physical_grasp & plausible & closure & ~m["open"]
    return old, shadow, d


def longest_run(flags):
    best = current = 0
    for value in torch.as_tensor(flags).bool().tolist():
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def phase_bounds(sequence=SEQUENCE, control_dt=.05):
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("invalid control timestep")
    result, step = {}, 0
    for name, count, _ in sequence:
        if name in result or not isinstance(count, int) or count <= 0:
            raise ValueError("unique phases and positive integer steps required")
        result[name] = (step * control_dt, (step + count) * control_dt)
        step += count
    return result


def summarize_calibration(trace, c=CONTRACT):
    f, dt = trace["frames"], trace["sample_dt_s"]
    sources = trace["sources"]
    if set(sources) != {1004, 1009, 1015, 1020, 1031} or len(sources) != 5:
        raise ValueError("all five unique TRAIN sources required")
    if trace.get("probe_sequence") != SEQUENCE:
        raise ValueError("calibration sequence mismatch")
    if not math.isclose(trace["control_dt_s"], .05) or not math.isclose(dt, .01):
        raise ValueError("M30 calibration requires 10ms physics and 20Hz control")
    bounds = phase_bounds()
    expected = round(bounds["open_again"][1] / dt) + 1
    if len(f["pos"]) != expected or not bool(f["active"].all()):
        raise ValueError("incomplete calibration or masked sources")
    m = {k: f[v].flatten(0, 1) for k, v in (
        ("red", "pos"), ("left_tip", "left_tip"), ("right_tip", "right_tip"),
        ("force", "force"), ("grip", "grip"), ("open", "open"))}
    old, shadow, d = shadow_grasp(m, c)
    shape = f["held"].shape
    old, shadow = old.reshape(shape), shadow.reshape(shape)
    if not torch.equal(old, f["held"]):
        raise ValueError("original held trace differs from recomputation")
    t = torch.arange(len(f["pos"]), dtype=torch.float64) * dt
    control = torch.arange(len(t)) % 5 == 0
    positive = (t >= .5) & (t <= bounds["closed_settle"][1])
    quiet = (t > bounds["closed_settle"][1] - 1) & (t <= bounds["closed_settle"][1])
    fd = angular_velocity_fd(f["quat_xyzw"][:-1], f["quat_xyzw"][1:], dt).norm(dim=-1)
    rows = []
    for i, source in enumerate(sources):
        movements = {}
        for name, axis, direction in (("small_lift", 2, 1), ("x_positive", 0, 1),
                ("x_negative", 0, -1), ("y_positive", 1, 1), ("y_negative", 1, -1)):
            a, b = [round(x / dt) for x in bounds[name]]
            red_delta = f["pos"][b, i] - f["pos"][a, i]
            ee_delta = f["ee"][b, i] - f["ee"][a, i]
            minimum = c.lift_min_m if axis == 2 else c.lateral_min_m
            error = float((red_delta - ee_delta).norm())
            movements[name] = {"red_delta_m": red_delta.tolist(), "ee_delta_m": ee_delta.tolist(),
                "follow_error_m": error, "passed": bool(direction * red_delta[axis] > minimum
                    and direction * ee_delta[axis] > minimum and error < c.follow_error_m)}
        yaw_motion = {}
        for name in ("small_yaw", "yaw_reverse"):
            a, b = [round(x / dt) for x in bounds[name]]
            angle = float(rotation_vector_world(f["quat_xyzw"][a, i], f["quat_xyzw"][b, i]).norm())
            yaw_motion[name] = {"red_rotation_rad": angle, "motion_observed": angle > c.yaw_min_rad}
        negatives = {}
        for name in ("open_release", "empty_hold", "empty_lift", "open_again"):
            end = bounds[name][1]
            mask = (t > end - .5) & (t <= end)
            mid = (f["left_tip"][mask, i] + f["right_tip"][mask, i]) / 2
            clearance = (mid - f["pos"][mask, i]).norm(dim=-1)
            empty = name.startswith("empty")
            valid = bool((clearance > c.empty_clearance_m).all() and
                (f["grip"][mask, i] < c.empty_finger_max_m).all() and not f["open"][mask, i].any()) if empty else bool(f["open"][mask, i].all())
            negatives[name] = {"shadow_false_positive_frames": int(shadow[mask, i].sum()),
                "old_false_positive_frames": int(old[mask, i].sum()), "frames": int(mask.sum()),
                "valid_negative": valid, "min_red_tip_midpoint_distance_m": float(clearance.min()),
                "passed": valid and not bool(shadow[mask, i].any())}
        w, lin = f["omega"][:, i].norm(dim=-1), f["lin"][:, i].norm(dim=-1)
        frac = float(shadow[positive, i].float().mean())
        missing = longest_run(~shadow[positive & control, i])
        stable = shadow[:, i] & (w < c.max_angular_radps) & (lin < c.max_linear_mps)
        final_stable = bool(stable[quiet & control].all())
        pos_ok = frac >= c.positive_fraction and missing <= c.max_missing_controls
        movement_ok = all(r["passed"] for r in movements.values()) and all(r["motion_observed"] for r in yaw_motion.values())
        negative_ok = all(r["passed"] for r in negatives.values())
        rows.append({"source_seed": source, "shadow_held_fraction": frac,
            "old_held_fraction": float(old[positive, i].float().mean()), "max_missing_control_frames": missing,
            "old_disagreement_frames": int((shadow[positive, i] & ~old[positive, i]).sum()),
            "positive_grasp_passed": pos_ok, "movement_passed": movement_ok,
            "negative_controls_passed": negative_ok, "unchanged_velocity_limits_passed": final_stable,
            "quiet_angular_api_radps": quantiles(w[quiet]), "quiet_linear_api_mps": quantiles(lin[quiet]),
            "quiet_angular_pose_fd_radps": quantiles(fd[quiet[1:], i]),
            "quiet_shadow_stable_controls": int(stable[quiet & control].sum()),
            "quiet_control_frames": int((quiet & control).sum()),
            "movements": movements, "yaw_motion": yaw_motion, "negatives": negatives,
            "passed": pos_ok and movement_ok and negative_ok and final_stable})
    return {"contract": asdict(c), "rows": rows, "all_five_passed": all(r["passed"] for r in rows),
        "scope": "shadow calibration only, no task-success rescore or policy training", "task_criteria_changed": False}
