"""Versioned v5 physical-skill contracts and bounded reference actions.

This module deliberately contains no Isaac, PPO, vision, or language imports.
It is the executable contract used by smoke tests and by future per-skill
environments.  The reference controller is a development probe only; it is
never a deployed action source or a substitute for a StagePPO actor.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch


ACTION_ORDER = ("dx", "dy", "dz", "dyaw", "grip")
SKILL_SEQUENCE = (
    "REACH", "GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND",
    "RELEASE_STABILIZE", "RETREAT",
)


class V5Skill(str, Enum):
    REACH = "REACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    ALIGN = "ALIGN"
    DESCEND = "DESCEND"
    RELEASE_STABILIZE = "RELEASE_STABILIZE"
    RETREAT = "RETREAT"


@dataclass(frozen=True)
class SkillTolerances:
    reach_distance_m: float = 0.035
    grasp_distance_m: float = 0.022
    # v4 accepted trajectories place the fingertip midpoint about 10 mm
    # above the red-cube center before closing.  Keeping this explicit avoids
    # driving the fingertips through the cube/table while entering GRASP.
    pregrasp_tip_height_m: float = 0.010
    # The relative-pose IK fingertip frames rise slightly as the fingers
    # close.  Start GRASP from the safe 10 mm pregrasp, then target 7 mm so
    # the measured closed geometry remains inside the v4 8-10 mm window.
    grasp_contact_tip_height_m: float = 0.007
    pregrasp_tip_xy_m: float = 0.012
    pregrasp_tip_z_error_m: float = 0.003
    lift_height_m: float = 0.060
    transport_xy_m: float = 0.045
    align_xy_m: float = 0.010
    descend_xy_m: float = 0.012
    descend_z_m: float = 0.004
    release_xy_m: float = 0.040
    release_z_m: float = 0.010
    retreat_distance_m: float = 0.100
    retreat_height_m: float = 0.080
    speed_mps: float = 0.050
    angular_speed_radps: float = 1.0


@dataclass(frozen=True)
class SkillContract:
    skill: V5Skill
    version: str
    observation_version: str
    observation_dim: int
    action_interface: str = "stage_vla_v5.action.v1"
    action_dim: int = 5
    entrance: str = ""
    success: str = ""
    failure: str = ""
    timeout: str = "steps >= episode_steps and not success/failure"

    def validate(self) -> None:
        if self.action_dim != 5 or self.action_interface != "stage_vla_v5.action.v1":
            raise ValueError(f"{self.skill}: invalid action interface")
        if self.observation_dim < 1:
            raise ValueError(f"{self.skill}: invalid observation dimension")
        if not all((self.entrance, self.success, self.failure, self.timeout)):
            raise ValueError(f"{self.skill}: incomplete terminal contract")


def _contract(skill: V5Skill, *, observation_dim: int, entrance: str,
              success: str, failure: str) -> SkillContract:
    result = SkillContract(
        skill=skill,
        version=f"stageppo-{skill.value.lower()}-v5-state{observation_dim}-action5-v1",
        observation_version=f"v5-state{observation_dim}-v1",
        observation_dim=observation_dim,
        entrance=entrance,
        success=success,
        failure=failure,
    )
    result.validate()
    return result


CONTRACTS: Mapping[V5Skill, SkillContract] = {
    V5Skill.REACH: _contract(
        V5Skill.REACH, observation_dim=52,
        entrance="object detected with finite 3-D pose and ee-object distance <= 0.30m",
        success="open gripper; fingertip midpoint XY is aligned and its height is object center + pregrasp_tip_height_m",
        failure="object pose invalid or ee-object distance > 0.40m for 2 steps",
    ),
    V5Skill.GRASP: _contract(
        V5Skill.GRASP, observation_dim=52,
        entrance="REACH physical pregrasp geometry; object remains within 0.05m of end effector",
        success="3 mm contact micro-descent with closure; physical grasp predicate true for 3 consecutive steps",
        failure="object dropped, finger geometry invalid, or timeout",
    ),
    V5Skill.LIFT: _contract(
        V5Skill.LIFT, observation_dim=52,
        entrance="physical grasp predicate true",
        success="held object height increased by lift_height_m and grasp retained",
        failure="grasp lost or object height below entry height - 0.015m",
    ),
    V5Skill.TRANSPORT: _contract(
        V5Skill.TRANSPORT, observation_dim=52,
        entrance="LIFT success with held object and support pose valid",
        success="held object XY distance, pressure, linear/angular speed stable for stable_steps",
        failure="grasp lost, object falls, or XY distance > 0.30m",
    ),
    V5Skill.ALIGN: _contract(
        V5Skill.ALIGN, observation_dim=52,
        entrance="TRANSPORT success with held object above support",
        success="held, align XY/height band, low speed for stable_steps",
        failure="grasp lost, object too low, or support distance is far",
    ),
    V5Skill.DESCEND: _contract(
        V5Skill.DESCEND, observation_dim=52,
        entrance="ALIGN success without simulator reset",
        success="held, stack XY/Z tolerances met, low speed for stable_steps",
        failure="grasp lost, object too low, or XY distance is far",
    ),
    V5Skill.RELEASE_STABILIZE: _contract(
        V5Skill.RELEASE_STABILIZE, observation_dim=52,
        entrance="DESCEND success without simulator reset",
        success="stack geometry met, gripper physically open, stable for stable_steps",
        failure="stack breaks, object moves far, or timeout",
    ),
    V5Skill.RETREAT: _contract(
        V5Skill.RETREAT, observation_dim=52,
        entrance="RELEASE_STABILIZE success without simulator reset",
        success="end effector is clear of object in distance and height",
        failure="supporting stack is disturbed or timeout",
    ),
}


def get_contract(skill: V5Skill | str) -> SkillContract:
    return CONTRACTS[V5Skill(skill)]


def _vec(value, *, device=None, dtype=torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != 3:
        raise ValueError("positions must have shape [N,3]")
    return tensor


def _col(value, n: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(n)
    if tensor.shape != (n,):
        raise ValueError(f"expected scalar or [{n}] values")
    return tensor


def _state(state: Mapping[str, object]) -> dict[str, torch.Tensor]:
    required = ("ee",)
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"missing state fields: {missing}")
    ee = _vec(state["ee"])
    object_key = "object" if "object" in state else "red"
    support_key = "support" if "support" in state else "blue"
    missing_roles = [key for key in (object_key, support_key) if key not in state]
    if missing_roles:
        raise ValueError("missing state fields: object/support (legacy red/blue aliases accepted)")
    manipulated = _vec(state[object_key], device=ee.device)
    support = _vec(state[support_key], device=ee.device)
    if not (ee.shape == manipulated.shape == support.shape):
        raise ValueError("ee, object and support must have matching shapes")
    n = ee.shape[0]
    result = {
        "ee": ee,
        "object": manipulated,
        "support": support,
        # Compatibility aliases for existing environment adapters.
        "red": manipulated,
        "blue": support,
    }
    result["left_tip"] = _vec(state.get("left_tip", ee), device=ee.device)
    result["right_tip"] = _vec(state.get("right_tip", ee), device=ee.device)
    if not (result["left_tip"].shape == result["right_tip"].shape == ee.shape):
        raise ValueError("finger-tip positions must match ee shape")
    result["speed"] = _col(state.get("speed", 0.0), n, device=ee.device)
    result["angular_speed"] = _col(
        state.get("angular_speed", 0.0), n, device=ee.device
    )
    result["open"] = _col(state.get("open", True), n, device=ee.device, dtype=torch.bool)
    result["held"] = _col(state.get("held", False), n, device=ee.device, dtype=torch.bool)
    result["stack_height"] = _col(state.get("stack_height", 0.0468), n, device=ee.device)
    if "grasp_target" in state:
        result["grasp_target"] = _vec(state["grasp_target"], device=ee.device)
        if result["grasp_target"].shape != ee.shape:
            raise ValueError("grasp_target must match ee shape")
    else:
        result["grasp_target"] = manipulated
    return result


def skill_success(skill: V5Skill | str, state: Mapping[str, object],
                  *, tolerances: SkillTolerances = SkillTolerances(),
                  stable_count: int | torch.Tensor = 1,
                  stable_steps: int = 3) -> torch.Tensor:
    """Evaluate a vectorized success predicate for smoke and adapters."""
    skill = V5Skill(skill)
    m = _state(state)
    n = m["ee"].shape[0]
    stable = _col(stable_count, n, device=m["ee"].device, dtype=torch.long)
    ee_object = (m["ee"] - m["object"]).norm(dim=-1)
    object_support = m["object"] - m["support"]
    xy = object_support[:, :2].norm(dim=-1)
    z_error = (object_support[:, 2] - m["stack_height"]).abs()
    speed_ok = m["speed"] < tolerances.speed_mps
    angular_speed_ok = m["angular_speed"] < tolerances.angular_speed_radps
    if skill == V5Skill.REACH:
        tip_mid = (m["left_tip"] + m["right_tip"]) / 2
        tip_error = tip_mid - m["grasp_target"]
        return (tip_error[:, :2].norm(dim=-1) < tolerances.pregrasp_tip_xy_m) & (
            (tip_error[:, 2] - tolerances.pregrasp_tip_height_m).abs()
            < tolerances.pregrasp_tip_z_error_m
        ) & m["open"]
    if skill == V5Skill.GRASP:
        return m["held"] & (ee_object < tolerances.grasp_distance_m) & (stable >= stable_steps)
    if skill == V5Skill.LIFT:
        return m["held"] & (m["object"][:, 2] - m["support"][:, 2] > tolerances.lift_height_m)
    if skill == V5Skill.TRANSPORT:
        return (
            m["held"]
            & (xy < tolerances.transport_xy_m)
            & speed_ok
            & angular_speed_ok
            & (stable >= stable_steps)
        )
    if skill == V5Skill.ALIGN:
        return m["held"] & (xy < tolerances.align_xy_m) & (z_error < 0.015) & speed_ok & (stable >= stable_steps)
    if skill == V5Skill.DESCEND:
        return m["held"] & (xy < tolerances.descend_xy_m) & (z_error < tolerances.descend_z_m) & speed_ok & (stable >= stable_steps)
    if skill == V5Skill.RELEASE_STABILIZE:
        return (xy < tolerances.release_xy_m) & (z_error < tolerances.release_z_m) & m["open"] & speed_ok & (stable >= stable_steps)
    rel = m["ee"] - m["object"]
    return (rel.norm(dim=-1) >= tolerances.retreat_distance_m) & (rel[:, 2] >= tolerances.retreat_height_m)


def skill_failure(skill: V5Skill | str, state: Mapping[str, object],
                  *, tolerances: SkillTolerances = SkillTolerances()) -> torch.Tensor:
    """Evaluate conservative physical-failure predicates."""
    skill = V5Skill(skill)
    m = _state(state)
    ee_object = (m["ee"] - m["object"]).norm(dim=-1)
    object_support = m["object"] - m["support"]
    xy = object_support[:, :2].norm(dim=-1)
    too_low = object_support[:, 2] < 0.025
    if skill in (V5Skill.GRASP, V5Skill.LIFT, V5Skill.TRANSPORT, V5Skill.ALIGN, V5Skill.DESCEND):
        return (~m["held"]) | too_low | (xy > 0.30)
    if skill == V5Skill.REACH:
        return ee_object > 0.40
    if skill == V5Skill.RELEASE_STABILIZE:
        return too_low | (xy > 0.15)
    return too_low | (xy > 0.15)


def reference_action(skill: V5Skill | str, state: Mapping[str, object],
                     *, tolerances: SkillTolerances = SkillTolerances(),
                     translation_limit_m: float = 0.005,
                     yaw_limit_rad: float = 0.02) -> torch.Tensor:
    """Return a bounded coupled [dx,dy,dz,dyaw,grip] development action."""
    if translation_limit_m <= 0 or yaw_limit_rad <= 0:
        raise ValueError("action limits must be positive")
    skill = V5Skill(skill)
    m = _state(state)
    n = m["ee"].shape[0]
    target = m["ee"].clone()
    grip = torch.ones(n, device=target.device)
    if skill in (V5Skill.REACH, V5Skill.GRASP):
        # The hand frame is offset from the actual finger contact midpoint.
        # v4's accepted physical traces use a pre-grasp fingertip midpoint
        # one centimetre above the cube center. GRASP then makes a small
        # contact-depth correction to compensate for closed-finger kinematics.
        tip_mid = (m["left_tip"] + m["right_tip"]) / 2
        desired_tip_mid = m["grasp_target"].clone()
        desired_tip_mid[:, 2] += (
            tolerances.pregrasp_tip_height_m if skill == V5Skill.REACH
            else tolerances.grasp_contact_tip_height_m
        )
        target = m["ee"] + (desired_tip_mid - tip_mid)
        grip = torch.ones(n, device=target.device) if skill == V5Skill.REACH else -torch.ones(n, device=target.device)
    elif skill == V5Skill.LIFT:
        target[:, 2] += tolerances.lift_height_m
        grip.fill_(-1)
    elif skill == V5Skill.TRANSPORT:
        target[:, :2] = m["support"][:, :2]
        grip.fill_(-1)
    elif skill == V5Skill.ALIGN:
        target[:, :2] = m["support"][:, :2]
        target[:, 2] = m["support"][:, 2] + m["stack_height"] + 0.012
        grip.fill_(-1)
        # Once the carried cube is geometrically aligned, stop issuing a
        # moving target and let the contact dynamics settle before handoff.
        rel = m["object"] - m["support"]
        aligned = (rel[:, :2].norm(dim=-1) < tolerances.align_xy_m) & (
            (rel[:, 2] - m["stack_height"]).abs() < 0.015
        )
        target = torch.where(aligned[:, None], m["ee"], target)
    elif skill == V5Skill.DESCEND:
        target = m["support"].clone()
        target[:, 2] += m["stack_height"]
        grip.fill_(-1)
        rel = m["object"] - m["support"]
        descended = (rel[:, :2].norm(dim=-1) < tolerances.descend_xy_m) & (
            (rel[:, 2] - m["stack_height"]).abs() < tolerances.descend_z_m
        )
        target = torch.where(descended[:, None], m["ee"], target)
    elif skill == V5Skill.RELEASE_STABILIZE:
        grip.fill_(1)
    elif skill == V5Skill.RETREAT:
        target[:, 2] += tolerances.retreat_height_m
        grip.fill_(1)
    delta = (target - m["ee"]).clamp(-translation_limit_m, translation_limit_m) / translation_limit_m
    action = torch.zeros((n, 5), device=target.device, dtype=target.dtype)
    action[:, :3] = delta
    action[:, 4] = grip
    return action.clamp(-1, 1)
