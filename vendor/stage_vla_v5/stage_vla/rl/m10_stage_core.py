"""Pure helpers for M10 stage-conditioned Action-DSL PPO.

This module intentionally has no Isaac Lab imports so the stage observation
encoding can be locked by ordinary unit tests.
"""

from __future__ import annotations

import torch
from torch import Tensor

from stage_vla.stages.stage_progress import ManipulationStage

M10_STAGE_COUNT = len(ManipulationStage)


def stage_one_hot(stage: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Encode stage ids as a 5-D one-hot tensor.

    Args:
        stage: Integer tensor of arbitrary batch shape containing valid
            ``ManipulationStage`` ids.
        dtype: Floating output dtype.
    """
    stage_t = torch.as_tensor(stage)
    if stage_t.dtype is not torch.long:
        raise TypeError(f"stage must be torch.long, got {stage_t.dtype}")
    valid = (stage_t >= int(ManipulationStage.REACH)) & (stage_t <= int(ManipulationStage.PLACE))
    if not bool(valid.all().item()):
        raise ValueError(f"unknown stage ids: {stage_t[~valid].tolist()}")
    return torch.nn.functional.one_hot(stage_t, num_classes=M10_STAGE_COUNT).to(dtype=dtype)


def append_stage_one_hot(policy_obs: Tensor, stage: Tensor) -> Tensor:
    """Append a stage one-hot vector to a flat policy observation batch."""
    obs = torch.as_tensor(policy_obs)
    stage_t = torch.as_tensor(stage, device=obs.device)
    if obs.ndim != 2:
        raise ValueError(f"policy_obs must be 2-D [N,D], got {tuple(obs.shape)}")
    if stage_t.shape != (obs.shape[0],):
        raise ValueError(
            f"stage must have shape ({obs.shape[0]},), got {tuple(stage_t.shape)}"
        )
    one_hot = stage_one_hot(stage_t, dtype=obs.dtype).to(device=obs.device)
    return torch.cat((obs, one_hot), dim=-1)
