"""Pure reset-distribution utilities for M17-A4 failure-driven curriculum.

No Isaac Lab dependency: this module only decides which reset mode to use and
samples target-XY offsets.  The vector environment owns actual snapshot restore.

Important experimental contract
-------------------------------
Held-out unseen seeds are NEVER loaded into training.  A4 only uses:
* the existing seed-1004 PLACE-entry snapshot for normal/edge recovery resets;
* DESCEND/SETTLE snapshots captured from the original five TRAIN seeds.
This keeps the 1100..1119 unseen split genuinely unseen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

NORMAL = 0
EDGE = 1
TERMINAL = 2
MODE_NAMES = ("normal", "edge", "terminal")


@dataclass(frozen=True)
class HardCaseCurriculumConfig:
    hard_fraction: float = 0.25
    terminal_fraction_within_hard: float = 0.50
    edge_min_fraction: float = 0.80
    terminal_xy_jitter_m: float = 0.005

    def validate(self) -> None:
        if not 0.0 <= self.hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in [0,1]")
        if not 0.0 <= self.terminal_fraction_within_hard <= 1.0:
            raise ValueError("terminal_fraction_within_hard must be in [0,1]")
        if not 0.0 <= self.edge_min_fraction <= 1.0:
            raise ValueError("edge_min_fraction must be in [0,1]")
        if self.terminal_xy_jitter_m < 0:
            raise ValueError("terminal_xy_jitter_m must be >= 0")


def assign_reset_modes(n: int, generator: torch.Generator, cfg: HardCaseCurriculumConfig) -> torch.Tensor:
    """Sample NORMAL / EDGE / TERMINAL mode ids on CPU."""
    cfg.validate()
    if n <= 0:
        return torch.zeros((0,), dtype=torch.long)
    u = torch.rand((n,), generator=generator)
    modes = torch.full((n,), NORMAL, dtype=torch.long)
    hard = u < cfg.hard_fraction
    if hard.any():
        v = torch.rand((int(hard.sum().item()),), generator=generator)
        hard_modes = torch.where(
            v < cfg.terminal_fraction_within_hard,
            torch.tensor(TERMINAL, dtype=torch.long),
            torch.tensor(EDGE, dtype=torch.long),
        )
        modes[hard] = hard_modes
    return modes


def sample_uniform_xy(n: int, range_m: float, generator: torch.Generator) -> torch.Tensor:
    if n <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    if range_m <= 0:
        return torch.zeros((n, 2), dtype=torch.float32)
    u = torch.rand((n, 2), generator=generator, dtype=torch.float32)
    return (u * 2.0 - 1.0) * float(range_m)


def sample_edge_xy(
    n: int,
    range_m: float,
    generator: torch.Generator,
    *,
    min_fraction: float = 0.80,
) -> torch.Tensor:
    """Sample offsets near the boundary of the already-validated +/-range box.

    At least one coordinate has |offset| in [min_fraction*range, range].  This
    strengthens recovery without widening the global training distribution to
    +/-5 cm (which A2 showed was worse).
    """
    if n <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    if range_m <= 0:
        return torch.zeros((n, 2), dtype=torch.float32)
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_fraction must be in [0,1]")
    out = sample_uniform_xy(n, range_m, generator)
    axis = torch.randint(0, 2, (n,), generator=generator)
    sign = torch.where(torch.rand((n,), generator=generator) < 0.5, -1.0, 1.0)
    mag = (min_fraction + (1.0 - min_fraction) * torch.rand((n,), generator=generator)) * float(range_m)
    out[torch.arange(n), axis] = sign * mag
    return out


def discover_terminal_snapshots(directory: str | Path) -> list[Path]:
    """Return only TRAIN-derived DESCEND/SETTLE snapshots by filename contract."""
    root = Path(directory)
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("*.pt")):
        name = p.stem.lower()
        if "_descend" in name or "_settle" in name:
            out.append(p)
    return out
