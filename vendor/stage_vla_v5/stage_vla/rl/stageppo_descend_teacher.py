"""Training-only guidance for independent StagePPO DESCEND demonstrations."""
from dataclasses import dataclass

import torch

from .stageppo_align_core import AlignConfig
from .stageppo_descend_core import DescendConfig, descend_errors


@dataclass(frozen=True)
class GuidedDescendConfig:
    align_xy_m: float = 0.008
    align_stable_steps: int = 3
    far_z_error_m: float = 0.008
    near_z_error_m: float = 0.004
    stop_z_error_m: float = 0.002
    far_z_action: float = -0.50
    near_z_action: float = -0.25
    contact_z_action: float = -0.10

    def validate(self, task_cfg):
        if not 0 < self.align_xy_m < task_cfg.xy_success_m:
            raise ValueError("guide alignment tolerance must lie inside DESCEND success tolerance")
        if not isinstance(self.align_stable_steps, int) or self.align_stable_steps < 1:
            raise ValueError("guide alignment stability must be a positive integer")
        if not 0 < self.stop_z_error_m < self.near_z_error_m < self.far_z_error_m:
            raise ValueError("guide Z thresholds must be strictly ordered")
        actions = (self.far_z_action, self.near_z_action, self.contact_z_action)
        if not all(-1 <= action < 0 for action in actions):
            raise ValueError("guide descent actions must lie in [-1, 0)")


def descend_to_align_observation(descend_obs, descend_cfg=None, align_cfg=None):
    """Re-express state52 using the frozen ALIGN actor's observation contract."""
    descend_cfg = descend_cfg or DescendConfig()
    align_cfg = align_cfg or AlignConfig()
    if descend_obs.ndim != 2 or descend_obs.shape[1] != 52:
        raise ValueError("DESCEND observation must be [N,52]")
    if not torch.isfinite(descend_obs).all():
        raise ValueError("DESCEND observation contains non-finite values")
    converted = descend_obs.clone()
    converted[:, 2] -= (align_cfg.height_target_m - descend_cfg.stack_height_m) / 0.05
    # The ALIGN actor never inherits DESCEND terminal bookkeeping. Preserve the
    # physical state and previous action, but normalize elapsed time to its own horizon.
    converted[:, 43:45] = 0.0
    converted[:, 45] *= descend_cfg.episode_steps / align_cfg.episode_steps
    return converted


def guided_descend_action(measured, align_action, descending, task_cfg=None, guide_cfg=None):
    """Compose training-only ALIGN guidance with a stateful vertical descent latch."""
    task_cfg = task_cfg or DescendConfig()
    guide_cfg = guide_cfg or GuidedDescendConfig()
    guide_cfg.validate(task_cfg)
    if align_action.ndim != 2 or align_action.shape[1] != 5:
        raise ValueError("ALIGN action must be [N,5]")
    if descending.shape != (len(align_action),) or descending.dtype != torch.bool:
        raise ValueError("descending latch must be a boolean vector")

    action = torch.zeros_like(align_action)
    align_cfg = AlignConfig()
    metric_ratio = align_cfg.translation_limit_m / task_cfg.translation_limit_m
    action[:, :2] = (align_action[:, :2] * metric_ratio).clamp(-1, 1)
    action[:, 3] = align_action[:, 3].clamp(-1, 1)
    action[:, 4] = -1.0

    _, height, z_error = descend_errors(measured, task_cfg)
    vertical = torch.zeros_like(height)
    above = height > task_cfg.stack_height_m
    vertical = torch.where(above & (z_error > guide_cfg.far_z_error_m),
                           vertical.new_full((), guide_cfg.far_z_action), vertical)
    vertical = torch.where(above & (z_error <= guide_cfg.far_z_error_m)
                           & (z_error > guide_cfg.near_z_error_m),
                           vertical.new_full((), guide_cfg.near_z_action), vertical)
    vertical = torch.where(above & (z_error <= guide_cfg.near_z_error_m)
                           & (z_error > guide_cfg.stop_z_error_m),
                           vertical.new_full((), guide_cfg.contact_z_action), vertical)
    action[descending, :2] = 0.0
    action[descending, 2] = vertical[descending]
    action[descending, 3] = 0.0
    settled = descending & (z_error <= guide_cfg.stop_z_error_m)
    return action, settled


def retargeted_align_action(align_action):
    """Use an ALIGN actor on a DESCEND-goal state; keep the training grasp closed."""
    if align_action.ndim != 2 or align_action.shape[1] != 5:
        raise ValueError("ALIGN action must be [N,5]")
    if not torch.isfinite(align_action).all():
        raise ValueError("ALIGN action contains non-finite values")
    action = align_action.clamp(-1, 1).clone()
    action[:, 4] = -1.0
    return action
