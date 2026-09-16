"""Collect optional per-step physical diagnostics for evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhysicalTraceCollector:
    args: Any
    env: Any
    helpers: Any
    rows: list[dict[str, object]] = field(default_factory=list)

    def append(
        self,
        *,
        relation_index: int,
        skill: str,
        step: int,
        checkpoint_action: Any = None,
        submitted_action: Any = None,
    ) -> None:
        if self.args.trace_env is None:
            return
        index = self.args.trace_env
        measured = self.env.measured
        relative = measured["red"][index] - measured["blue"][index]
        jaw = measured["right_tip"][index] - measured["left_tip"][index]
        leveling = self.helpers.jaw_leveling_axis_angle(
            measured["left_tip"][index:index + 1],
            measured["right_tip"][index:index + 1],
        )[0]
        row: dict[str, object] = {
            "relation": relation_index + 1,
            "skill": skill,
            "step": int(step),
            "projected_action": self.env.prev_unit[index].detach().cpu().tolist(),
            "stack_relative_xyz_m": relative.detach().cpu().tolist(),
            "object_quaternion_xyzw": (
                measured["red_quat"][index].detach().cpu().tolist()
            ),
            "upright_tilt_rad": float(self.helpers.object_upright_tilt_rad(
                measured["red_quat"][index:index + 1]
            )[0]),
            "angular_velocity_radps": (
                measured["angular"][index].detach().cpu().tolist()
            ),
            "angular_speed_radps": float(
                measured.get(
                    "stability_angular_speed", measured["angular"].norm(dim=-1)
                )[index]
            ),
            "instantaneous_angular_speed_radps": float(
                measured["angular"][index].norm()
            ),
            "control_angular_speed_radps": float(
                measured.get(
                    "control_angular_speed", measured["angular"].norm(dim=-1)
                )[index]
            ),
            "linear_velocity_mps": measured["vel"][index].detach().cpu().tolist(),
            "stability_speed_mps": float(measured["stability_speed"][index]),
            "instantaneous_speed_mps": float(measured["speed"][index]),
            "left_fingertip_position_m": (
                measured["left_tip"][index].detach().cpu().tolist()
            ),
            "right_fingertip_position_m": (
                measured["right_tip"][index].detach().cpu().tolist()
            ),
            "jaw_axis_w": jaw.detach().cpu().tolist(),
            "jaw_height_delta_m": float(jaw[2]),
            "jaw_leveling_axis_angle_rad": leveling.detach().cpu().tolist(),
            "force_n": measured["force"][index].detach().cpu().tolist(),
            "physical_grasp": bool(measured["physical"][index]),
            "stable_steps": int(self.env.stable_count[index]),
        }
        if checkpoint_action is not None:
            row["checkpoint_action"] = (
                checkpoint_action[index].detach().cpu().tolist()
            )
        if submitted_action is not None:
            row["submitted_action"] = (
                submitted_action[index].detach().cpu().tolist()
            )
        self.rows.append(row)


__all__ = ["PhysicalTraceCollector"]
