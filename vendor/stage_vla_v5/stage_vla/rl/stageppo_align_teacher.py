"""Demonstration teachers for isolated StagePPO ALIGN training."""
from __future__ import annotations

import torch


def settling_align_teacher_action(measured, cfg, clip, deadband_m=0.003):
    """Move to the hover goal, then command zero motion so the cube can settle."""
    if not 0 < deadband_m < cfg.xy_success_m:
        raise ValueError("settling deadband must lie inside the XY success tolerance")
    clip = torch.as_tensor(clip, device=measured["red"].device, dtype=measured["red"].dtype)
    if clip.shape != (4,) or (clip <= 0).any():
        raise ValueError("ALIGN teacher clip must contain four positive values")
    target = measured["blue"] + measured["blue"].new_tensor([0.0, 0.0, cfg.height_target_m])
    error = target - measured["red"]
    metric = error.clamp(-clip[:3], clip[:3])
    settled_position = (error[:, :2].norm(dim=-1) < deadband_m) & (error[:, 2].abs() < deadband_m)
    metric[settled_position] = 0.0
    action = measured["red"].new_zeros((len(error), 5))
    action[:, :3] = metric / cfg.translation_limit_m
    action[:, 4] = -1.0
    return action.clamp(-1, 1), settled_position
