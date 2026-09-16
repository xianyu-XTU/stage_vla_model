"""M16 PLACE state-based BC: two-head policy (continuous motion + discrete grip).

Continuous head: d(x,y,z,yaw)  (env delta actions)
Discrete head:   grip CLOSE/OPEN

Separating the two avoids the "MSE over averaged actions ~ 0" collapse: the
motion head never has to represent the grip OPEN, and the grip head is a clean
classifier. OPEN is latched at execution time (once commanded, keep open).

State (object-centric, blue-local frame), 25-d:
  red_rel_target(3), ee_rel_red(3), red_lin_vel(3), red_ang_vel(3),
  gripper_joint_pos(2), finger_forces(2), prev_action(4), micro_stage_onehot(5)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

MICRO_NAMES = ("ALIGN", "DESCEND", "SETTLE", "RELEASE", "FAIL")
STATE_DIM = 25  # 20 base + 5 micro-stage one-hot
CONT_DIM = 4  # dx, dy, dz, dyaw
NUM_GRIP = 2  # CLOSE=0, OPEN=1


@dataclass
class PlaceBCDataset:
    states: torch.Tensor
    cont: torch.Tensor
    grip: torch.Tensor
    micro: torch.Tensor

    def __len__(self) -> int:
        return len(self.states)


def load_place_bc(npz_dir: str | Path, seeds: list[int] | None = None) -> PlaceBCDataset:
    npz_dir = Path(npz_dir)
    files = [npz_dir / f"seed_{s}.npz" for s in seeds] if seeds else sorted(npz_dir.glob("seed_*.npz"))
    states, cont, grip, micro = [], [], [], []
    for f in files:
        if not f.is_file():
            print(f"[bc] skip missing {f}")
            continue
        d = np.load(f)
        states.append(np.asarray(d["states"], dtype=np.float32))
        cont.append(np.asarray(d["cont_action"], dtype=np.float32))
        micro_i = np.asarray(d["micro_stage"], dtype=np.int64)
        # D2.1 causal grip contract: RELEASE is the only OPEN target.  Older
        # recorded files may contain a few SETTLE+OPEN rows inherited from raw
        # expert lag; do not let those contradict the geometry-phase contract.
        grip.append((micro_i == 3).astype(np.int64))
        micro.append(micro_i)
    return PlaceBCDataset(
        states=torch.from_numpy(np.concatenate(states)),
        cont=torch.from_numpy(np.concatenate(cont)),
        grip=torch.from_numpy(np.concatenate(grip)),
        micro=torch.from_numpy(np.concatenate(micro)),
    )


def load_dagger(npz_path: str | Path) -> PlaceBCDataset:
    """Load a DAgger npz (states, expert_action, phase, recoverable).

    Accepts either the plain-round format (geometry_phase) or the staged format
    (phase_before). Converts the 7-D expert_action to the BC training format
    (cont dx,dy,dz,dyaw + grip CLOSE/OPEN label).
    """
    d = np.load(npz_path)
    states = np.asarray(d["states"], dtype=np.float32)
    ex = np.asarray(d["expert_action"], dtype=np.float32)
    phase = np.asarray(d.get("phase_before", d.get("geometry_phase")), dtype=np.int64)
    recoverable = np.asarray(d["recoverable"], dtype=np.int64)
    if len(states) == 0:
        return PlaceBCDataset(torch.zeros(0, 25), torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    keep = recoverable == 1
    return PlaceBCDataset(
        states=torch.from_numpy(states[keep]),
        cont=torch.from_numpy(ex[keep][:, [0, 1, 2, 5]]),
        grip=torch.from_numpy((ex[keep][:, 6] >= 0).astype(np.int64)),
        micro=torch.from_numpy(phase[keep]),
    )


class StageBalancedSampler(torch.utils.data.Sampler):
    """Batch-yielding sampler: each yielded item is a full batch with fixed
    counts per micro-stage, so ALIGN cannot drown DESCEND/SETTLE/RELEASE.

    Used with DataLoader(batch_size=None). Yields exactly ``num_batches`` batches.
    """

    def __init__(self, micro: torch.Tensor, stages_per_batch: dict[int, int],
                 num_batches: int, num_stages: int = 5, seed: int = 0):
        self.micro = micro
        self.stages_per_batch = stages_per_batch
        self.num_batches = num_batches
        self.num_stages = num_stages
        self.g = torch.Generator().manual_seed(seed)
        self.per_stage_idx = [torch.nonzero(micro == i, as_tuple=False).flatten() for i in range(num_stages)]
        self.batch_size = sum(self.stages_per_batch.values())

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for stage, k in self.stages_per_batch.items():
                idx = self.per_stage_idx[stage]
                if len(idx) == 0:
                    continue
                pick = idx[torch.randint(0, len(idx), (k,), generator=self.g)]
                batch.append(pick)
            if not batch:
                return
            yield torch.cat(batch).tolist()

    def __len__(self):
        return self.num_batches


def merge_datasets(*datasets: PlaceBCDataset) -> PlaceBCDataset:
    return PlaceBCDataset(
        states=torch.cat([d.states for d in datasets]),
        cont=torch.cat([d.cont for d in datasets]),
        grip=torch.cat([d.grip for d in datasets]),
        micro=torch.cat([d.micro for d in datasets]),
    )


class PlaceBC(nn.Module):
    """State -> (continuous motion delta, discrete grip).

    The continuous head predicts in NORMALIZED units (z-score) so the many small
    expert per-step deltas do not dominate the MSE and collapse the output toward
    the mean ("average action ~ 0"). ``act()`` denormalizes back to env units
    using the stored cont_mean / cont_std.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.cont_head = nn.Linear(hidden, CONT_DIM)
        self.grip_head = nn.Linear(hidden, NUM_GRIP)
        self.register_buffer("cont_mean", torch.zeros(CONT_DIM))
        self.register_buffer("cont_std", torch.ones(CONT_DIM))

    def set_cont_norm(self, mean, std):
        self.cont_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.cont_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, state):
        h = self.encoder(state)
        return self.cont_head(h), self.grip_head(h)

    def act(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic: (denormalized cont_delta(4), grip_label)."""
        self.eval()
        with torch.inference_mode():
            c, g = self.forward(state)
        return c * self.cont_std + self.cont_mean, g.argmax(-1)


def normalize_cont(ds: PlaceBCDataset, mean, std) -> PlaceBCDataset:
    return PlaceBCDataset(
        states=ds.states,
        cont=(ds.cont - torch.as_tensor(mean, dtype=torch.float32)) / torch.as_tensor(std, dtype=torch.float32),
        grip=ds.grip,
        micro=ds.micro,
    )


def cont_stats(ds: PlaceBCDataset):
    mean = ds.cont.mean(0)
    std = ds.cont.std(0) + 1e-6
    return mean, std


def micro_stage_weights(ds: PlaceBCDataset) -> torch.Tensor:
    """Inverse-frequency weights per micro-stage for balanced sampling."""
    counts = torch.bincount(ds.micro, minlength=len(MICRO_NAMES)).to(torch.float32)
    w = counts.sum() / counts.clamp(min=1)
    return w / w.sum()


def bc_loss(cont_pred, grip_pred, cont, grip, grip_weight: float = 1.0,
            mag_weight: float = 0.0):
    """Continuous + discrete BC loss.

    ``mag_weight > 0`` weights each sample's cont MSE by its target action
    magnitude, so the many small per-step deltas do not dominate the mean and the
    policy learns to output the full-range expert actions (fixes "average ~ 0").
    """
    if mag_weight > 0:
        w = 1.0 + mag_weight * cont.norm(dim=-1, keepdim=True)
        l_cont = (w * (cont_pred - cont).pow(2)).mean()
    else:
        l_cont = F.mse_loss(cont_pred, cont)
    l_grip = F.cross_entropy(grip_pred, grip)
    return {"cont": l_cont, "grip": l_grip, "total": l_cont + grip_weight * l_grip}



def natural_loader(
    ds: PlaceBCDataset, batch_size: int, num_workers: int = 0, shuffle: bool = True
):
    """Ordinary empirical-distribution loader (no inverse-stage oversampling).

    Use this for D2.1 when newly collected terminal DAgger data should remain a
    minority correction instead of being amplified to equal stage frequency.
    """
    from torch.utils.data import TensorDataset, DataLoader

    tds = TensorDataset(ds.states, ds.cont, ds.grip, ds.micro)
    return DataLoader(tds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def balanced_loader(
    ds: PlaceBCDataset, batch_size: int, num_workers: int = 0, shuffle: bool = True
):
    from torch.utils.data import TensorDataset, DataLoader

    weights = micro_stage_weights(ds)[ds.micro]
    tds = TensorDataset(ds.states, ds.cont, ds.grip, ds.micro)
    if shuffle:
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(tds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return DataLoader(tds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def stage_balanced_loader(
    ds: PlaceBCDataset,
    stages_per_batch: dict[int, int],
    num_batches: int,
    num_workers: int = 0,
    seed: int = 0,
):
    """Each yielded batch has fixed counts per micro-stage; batch_size=None so
    the sampler's full batch is used as-is."""
    from torch.utils.data import TensorDataset, DataLoader

    sampler = StageBalancedSampler(ds.micro, stages_per_batch, num_batches, seed=seed)
    tds = TensorDataset(ds.states, ds.cont, ds.grip, ds.micro)
    return DataLoader(tds, batch_size=None, sampler=sampler, num_workers=num_workers)
