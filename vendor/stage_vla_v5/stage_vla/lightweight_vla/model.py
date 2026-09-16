"""M16 Stage-aware Lightweight VLA Student model (v2, improved training signal).

Key changes vs v1 (which collapsed to constant actions):
- xyz/rot predicted in *normalized* units (per-dim z-score) so the tiny expert
  per-step deltas are not noise-dominated in MSE.
- gripper is treated as a *binary classification* (open/close) instead of
  regressing {-1,1} -- regressing binary grip collapses to ~0.
- stage CE is class-weighted (REACH dominates 55%).

Pipeline: image 128x128x3 --VisionEncoder--> v ; instruction --InstructionEncoder--> i
  f = MLP([v, i]) ; stage_logits = StageHead(f) ; action_xyzrot = ActionHead([f, stage_emb])
  grip_logit = GripHead([f, stage_emb]) ; decode -> 7-D env action.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTION_DIM = 7
STAGES = ("REACH", "GRASP", "LIFT", "TRANSPORT", "PLACE")
NUM_STAGES = len(STAGES)


class VisionEncoder(nn.Module):
    """Tiny CNN: 128x128x3 -> 256-d visual feature."""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(256, feat_dim), nn.ReLU(inplace=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.convs(image))


class InstructionEncoder(nn.Module):
    """Tiny bag-of-words encoder: instruction str -> 32-d feature."""

    def __init__(self, vocab: list[str], feat_dim: int = 32):
        super().__init__()
        self.vocab = list(vocab)
        self.word2id = {w: i + 2 for i, w in enumerate(self.vocab)}  # 0=PAD, 1=UNK
        self.embed = nn.Embedding(len(self.vocab) + 2, 16, padding_idx=0)
        self.proj = nn.Linear(16, feat_dim)

    def _ids(self, text: str) -> list[int]:
        return [self.word2id.get(w, 1) for w in text.lower().split()] or [1]

    def forward(self, text: list[str]) -> torch.Tensor:
        ids = [self._ids(t) for t in text]
        B, L = len(ids), max(len(x) for x in ids)
        idx = torch.zeros(B, L, dtype=torch.long, device=next(self.parameters()).device)
        for b, row in enumerate(ids):
            idx[b, : len(row)] = torch.tensor(row, dtype=torch.long)
        emb = self.embed(idx).mean(dim=1)
        return F.relu(self.proj(emb))


class StageAwareVLA(nn.Module):
    """image + instruction -> stage_logits(5) + xyzrot_norm(6) + grip_logit(2).

    Normalization stats (xyz_mean, xyz_std of the 6-D xyz/rot) must be provided
    so decode_action() can map predictions back to env action units.
    """

    def __init__(
        self,
        vocab: list[str],
        xyz_mean: np.ndarray,
        xyz_std: np.ndarray,
        *,
        vis_feat: int = 256,
        inst_feat: int = 32,
        stage_emb: int = 32,
        hidden: int = 256,
    ):
        super().__init__()
        self.register_buffer("xyz_mean", torch.tensor(np.asarray(xyz_mean, np.float32)))
        self.register_buffer("xyz_std", torch.tensor(np.asarray(xyz_std, np.float32)))
        self.vision = VisionEncoder(vis_feat)
        self.instr = InstructionEncoder(vocab, inst_feat)
        self.fusion = nn.Sequential(nn.Linear(vis_feat + inst_feat, hidden), nn.ReLU(inplace=True))
        self.stage_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, NUM_STAGES))
        self.stage_embed = nn.Embedding(NUM_STAGES, stage_emb)
        head_in = hidden + stage_emb
        self.action_head = nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 6))
        self.grip_head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, image, instruction, stage=None):
        v = self.vision(image)
        i = self.instr(instruction)
        f = self.fusion(torch.cat([v, i], dim=-1))
        stage_logits = self.stage_head(f)
        if stage is not None:
            stage_ids = stage  # teacher forcing at train time
        else:
            stage_ids = stage_logits.argmax(dim=-1)
        se = self.stage_embed(stage_ids)
        head_in = torch.cat([f, se], dim=-1)
        xyzrot = self.action_head(head_in)
        grip_logit = self.grip_head(head_in)
        return stage_logits, xyzrot, grip_logit

    def decode_action(self, xyzrot: torch.Tensor, grip_logit: torch.Tensor) -> torch.Tensor:
        """Map normalized model outputs to a 7-D env action."""
        xyzrot_d = xyzrot * self.xyz_std + self.xyz_mean
        grip = torch.where(grip_logit.argmax(-1, keepdim=True) > 0,
                           torch.tensor(1.0), torch.tensor(-1.0))
        return torch.cat([xyzrot_d, grip], dim=-1)

    def act(self, image, instruction) -> torch.Tensor:
        stage_logits, xyzrot, grip_logit = self.forward(image, instruction, stage=None)
        return self.decode_action(xyzrot, grip_logit)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def stage_class_weights(stage_counts: dict[int, int]) -> torch.Tensor:
    """Inverse-frequency class weights for the imbalanced stage labels."""
    total = sum(stage_counts.values())
    w = torch.tensor([total / max(stage_counts.get(i, 0), 1) for i in range(NUM_STAGES)], dtype=torch.float32)
    return w / w.sum()


def m16_loss(
    stage_logits, xyzrot, grip_logit, *,
    stage, expert_xyzrot_norm, grip_binary, openvla_xyz_norm,
    stage_weight=0.5, openvla_weight=0.1, stage_w=None,
) -> dict[str, torch.Tensor]:
    loss_stage = F.cross_entropy(stage_logits, stage.long(), weight=stage_w)
    loss_xyz = F.mse_loss(xyzrot, expert_xyzrot_norm)
    loss_grip = F.cross_entropy(grip_logit, grip_binary.long())
    loss_expert = loss_xyz + loss_grip
    out = {"stage": loss_stage, "xyz": loss_xyz, "grip": loss_grip,
           "expert": loss_expert, "total": loss_expert + stage_weight * loss_stage}
    if openvla_xyz_norm is not None:
        loss_ov = F.mse_loss(xyzrot[:, :3], openvla_xyz_norm)
        out["openvla"] = loss_ov
        out["total"] = out["total"] + openvla_weight * loss_ov
    return out
