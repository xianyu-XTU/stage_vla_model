"""Focused ALIGN curriculum with a frozen-policy anchor on retained sources."""
from __future__ import annotations

import torch

from .stageppo_align_recovery_vecenv import FocusedAlignV2VecEnv


def teacher_anchor_penalty(
    actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    source_indices: torch.Tensor,
    focus_index: int,
    active: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    if actions.shape != teacher_actions.shape or actions.ndim != 2:
        raise ValueError("student and teacher action tensors must have the same [N,A] shape")
    if source_indices.shape != active.shape or len(source_indices) != len(actions):
        raise ValueError("source and active masks must match the action batch")
    if coefficient < 0:
        raise ValueError("teacher anchor coefficient must be nonnegative")
    anchor = (source_indices != focus_index) & active
    mismatch = (actions.clamp(-1, 1) - teacher_actions.clamp(-1, 1)).square().mean(dim=-1)
    return coefficient * mismatch * anchor.float()


class TeacherAnchoredFocusedAlignV2VecEnv(FocusedAlignV2VecEnv):
    """Let PPO adapt the focus source while retaining frozen anchor behavior."""

    def __init__(self, env, *, teacher_policy, teacher_anchor_coefficient=0.2, **kwargs):
        super().__init__(env, **kwargs)
        if teacher_policy is None or teacher_anchor_coefficient < 0:
            raise ValueError("a valid frozen teacher and coefficient are required")
        self.teacher_policy = teacher_policy
        self.teacher_anchor_coefficient = float(teacher_anchor_coefficient)
        self.focus_source_index = self.source_seeds.index(self.focus_source_seed)
        self.interface = {
            **self.interface,
            "teacher_anchor": {
                "training_only": True,
                "focus_source_seed": self.focus_source_seed,
                "coefficient": self.teacher_anchor_coefficient,
                "deployment_dependency": False,
            },
        }

    def step(self, actions):
        active = ~self.finished
        student = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            teacher = self.teacher_policy(self._obs()["policy"])
        penalty = teacher_anchor_penalty(
            student,
            teacher,
            self.last_source_idx,
            self.focus_source_index,
            active,
            self.teacher_anchor_coefficient,
        )
        obs, reward, done, extras = super().step(student)
        reward = reward - penalty
        self.returns -= penalty
        extras["log"]["align_v2/teacher_anchor_penalty"] = penalty.mean()
        return obs, reward, done, extras
