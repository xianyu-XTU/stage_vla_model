"""Frozen 52-dimensional observation contract for the cube REACH policy."""

from __future__ import annotations

import torch


REACH_OBS_DIM = 52


def reach_observation(
    state: dict[str, torch.Tensor], previous_action: torch.Tensor
) -> torch.Tensor:
    """Encode the checkpoint-compatible REACH observation."""
    count = state["ee"].shape[0]
    target = torch.as_tensor(
        state.get("grasp_target", state["red"]),
        device=state["ee"].device,
        dtype=state["ee"].dtype,
    )
    if target.shape != state["red"].shape:
        raise ValueError("grasp_target must match object position shape")
    previous_action = torch.as_tensor(
        previous_action,
        device=state["ee"].device,
        dtype=state["ee"].dtype,
    )
    if previous_action.shape != (count, 5):
        raise ValueError("previous_action must have shape [N,5]")
    observation = torch.cat(
        [
            (state["red"] - state["blue"]) / 0.5,
            (state["ee"] - target) / 0.5,
            (state["left_tip"] - target) / 0.1,
            (state["right_tip"] - target) / 0.1,
            state["red_quat"],
            state["blue_quat"],
            state["q"] / 3.0,
            state["qd"] / 2.0,
            state["grip"] / 0.04,
            previous_action,
            state["speed"].unsqueeze(-1) / 0.2,
            state["ee"],
            state["red_vel"] / 0.2,
            state["red_ang"] / 2.0,
            state["open"].float().unsqueeze(-1),
        ],
        dim=-1,
    )
    if observation.shape != (count, REACH_OBS_DIM) or not torch.isfinite(
        observation
    ).all():
        raise RuntimeError("invalid REACH observation")
    return observation.clamp(-10.0, 10.0)
