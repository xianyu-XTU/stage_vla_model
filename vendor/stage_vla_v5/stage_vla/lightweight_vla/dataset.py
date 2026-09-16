"""M16 dataset: load M15 distillation npz into a torch Dataset for Student BC.

Provides normalized 6-D xyz/rot targets (z-score), binary gripper labels, and
normalized OpenVLA xyz (auxiliary), so the tiny expert per-step deltas are not
noise-dominated in MSE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_m16_frames(npz_dir: str | Path, seeds: list[int] | None = None):
    """Concatenate M15 npz frames into arrays. Returns a dict of arrays."""
    npz_dir = Path(npz_dir)
    if seeds is None:
        files = sorted(npz_dir.glob("seed_*.npz"))
    else:
        files = [npz_dir / f"seed_{s}.npz" for s in seeds]
    images, stages, expert, openvla, instruction = [], [], [], [], None
    for f in files:
        if not f.is_file():
            print(f"[m16] skip missing {f}")
            continue
        d = np.load(f, allow_pickle=True)
        images.append(np.asarray(d["images"]))
        stages.append(np.asarray(d["stage"], dtype=np.int64))
        expert.append(np.asarray(d["expert_action"], dtype=np.float32))
        openvla.append(np.asarray(d["openvla_action"], dtype=np.float32))
        inst = str(np.asarray(d["instruction"]).item())
        if instruction is None:
            instruction = inst
    return {
        "images": np.concatenate(images) if images else np.zeros((0, 128, 128, 3), np.uint8),
        "stage": np.concatenate(stages) if stages else np.zeros((0,), np.int64),
        "expert": np.concatenate(expert) if expert else np.zeros((0, 7), np.float32),
        "openvla": np.concatenate(openvla) if openvla else np.zeros((0, 7), np.float32),
        "instruction": instruction or "",
    }


class M16Dataset(Dataset):
    """Frame-wise BC dataset with normalized targets."""

    def __init__(self, data: dict):
        self.images = data["images"]          # (N,128,128,3) uint8
        self.stage = data["stage"]            # (N,)
        expert = data["expert"]               # (N,7)
        self.openvla = data["openvla"]        # (N,7)
        self.instruction = data["instruction"]
        assert len(self.images) == len(self.stage) == len(expert) == len(self.openvla)

        # normalization stats on the 6-D xyz/rot (expert is the primary target)
        self.xyz_mean = expert[:, :6].mean(0)
        self.xyz_std = expert[:, :6].std(0) + 1e-6
        self.expert_xyz_norm = (expert[:, :6] - self.xyz_mean) / self.xyz_std   # (N,6)
        self.grip_binary = (expert[:, 6] > 0).astype(np.int64)                  # 1=open
        # OpenVLA auxiliary: normalize its xyz with the same stats
        self.openvla_xyz_norm = (self.openvla[:, :3] - self.xyz_mean[:3]) / self.xyz_std[:3]

    def stats(self):
        return {"xyz_mean": self.xyz_mean, "xyz_std": self.xyz_std}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).to(torch.float32) / 255.0 - 0.5
        return {
            "image": img,
            "instruction": self.instruction,
            "stage": torch.tensor(int(self.stage[i]), dtype=torch.long),
            "expert_xyz_norm": torch.from_numpy(self.expert_xyz_norm[i]),
            "grip_binary": torch.tensor(int(self.grip_binary[i]), dtype=torch.long),
            "openvla_xyz_norm": torch.from_numpy(self.openvla_xyz_norm[i]),
        }
