"""Frozen known-size cube-policy observation construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stage_vla_v7.interfaces import PhysicalState


LEGACY_OBSERVATION_DIM = 55
PHYSICAL_CONTEXT_DIM = 10


@dataclass(frozen=True)
class KnownSizeObservationInput:
    state: PhysicalState
    target_force_n: object
    max_force_n: float
    size_features: object
    previous_action: object
    previous_force: object
    stable_count: object
    stable_steps: int
    step_count: object
    episode_steps: int
    step_dt_s: float
    visual_object_position: object | None = None
    visual_support_position: object | None = None
    physical_context: object | None = None


def build_known_size_observation(inputs: KnownSizeObservationInput) -> Any:
    """Build the exact legacy 55-D observation without simulator access."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("known-size observations require the 'torch' extra") from exc

    if inputs.stable_steps < 1 or inputs.episode_steps < 1:
        raise ValueError("stable_steps and episode_steps must be positive")
    if inputs.step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be positive")
    if inputs.max_force_n <= 0.0:
        raise ValueError("max_force_n must be positive")
    if (inputs.visual_object_position is None) != (
        inputs.visual_support_position is None
    ):
        raise ValueError("visual object and support positions must be supplied together")

    state = inputs.state
    force = torch.as_tensor(state.fingertip_force)
    target_force = torch.as_tensor(
        inputs.target_force_n, device=force.device, dtype=force.dtype
    ).reshape(-1)
    previous_force = torch.as_tensor(
        inputs.previous_force, device=force.device, dtype=force.dtype
    )
    object_position = torch.as_tensor(state.object_position)
    support_position = torch.as_tensor(
        state.support_position,
        device=object_position.device,
        dtype=object_position.dtype,
    )
    if inputs.visual_object_position is not None:
        object_position = torch.as_tensor(
            inputs.visual_object_position,
            device=object_position.device,
            dtype=object_position.dtype,
        )
        support_position = torch.as_tensor(
            inputs.visual_support_position,
            device=object_position.device,
            dtype=object_position.dtype,
        )

    count = object_position.shape[0]
    if object_position.shape != (count, 3) or support_position.shape != (count, 3):
        raise ValueError("object and support positions must have shape [N,3]")
    if force.shape != (count, 2) or previous_force.shape != force.shape:
        raise ValueError("force and previous_force must have shape [N,2]")
    if target_force.shape != (count,):
        raise ValueError("target_force_n must have shape [N]")

    grip = torch.as_tensor(state.gripper_joint_position)
    force_rate = (force - previous_force) / float(inputs.step_dt_s)
    fields = [
        (object_position - support_position) / 0.05,
        (torch.as_tensor(state.end_effector_position) - object_position) / 0.05,
        torch.as_tensor(state.object_linear_velocity) / 0.2,
        torch.as_tensor(state.object_angular_velocity) / 2.0,
        torch.as_tensor(state.object_orientation),
        torch.as_tensor(state.support_orientation),
        grip / 0.04,
        force / 10.0,
        torch.as_tensor(state.arm_joint_position) / 3.0,
        torch.as_tensor(state.arm_joint_velocity) / 2.0,
        torch.as_tensor(inputs.previous_action),
        (torch.as_tensor(inputs.stable_count) / inputs.stable_steps).unsqueeze(-1),
        (torch.as_tensor(inputs.step_count) / inputs.episode_steps).unsqueeze(-1),
        torch.as_tensor(inputs.size_features),
        grip.sum(dim=-1, keepdim=True) / 0.08,
        force.mean(dim=-1, keepdim=True) / 10.0,
        (force[:, :1] - force[:, 1:2]) / 10.0,
        force_rate / 20.0,
        target_force.unsqueeze(-1) / inputs.max_force_n,
        torch.as_tensor(state.physical_grasp).float().unsqueeze(-1),
    ]
    if inputs.physical_context is not None:
        fields.append(torch.as_tensor(inputs.physical_context))
    observation = torch.cat(fields, dim=-1)
    expected_dim = LEGACY_OBSERVATION_DIM + (
        PHYSICAL_CONTEXT_DIM if inputs.physical_context is not None else 0
    )
    if observation.shape != (count, expected_dim):
        raise RuntimeError(
            f"invalid known-size observation shape {tuple(observation.shape)}; "
            f"expected {(count, expected_dim)}"
        )
    if not torch.isfinite(observation).all():
        raise RuntimeError("known-size observation contains non-finite values")
    return observation.clamp(-10.0, 10.0)
