"""M16 DAgger expert contract helpers.

This module keeps DAgger supervision consistent with the causal geometry phase
tracker and calibrates expert motion magnitudes from the original successful
M16 PLACE dataset instead of using a fixed 0.05 action clip.

Contract:
  ALIGN/DESCEND/SETTLE -> gripper CLOSE
  RELEASE              -> gripper OPEN
  FAIL                 -> no-op CLOSE (collector should terminate instead)

The continuous correction is a small phase-aware proportional correction whose
per-axis magnitude is clipped to a percentile measured from the original expert
PLACE data.  This prevents DAgger from teaching much larger actions than the
successful demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry_phase import ALIGN, DESCEND, FAIL, MICRO_NAMES, RELEASE, SETTLE

CONT_IDXS = np.asarray([0, 1, 2, 5], dtype=np.int64)  # dx,dy,dz,dyaw in raw 7-D action
NUM_PHASES = len(MICRO_NAMES)
DEFAULT_STACK_HEIGHT_DIFF_M = 0.0468


@dataclass(frozen=True)
class ExpertActionProfile:
    """Robust per-phase continuous action limits from the original dataset."""

    abs_clip: np.ndarray  # [phase, 4] pXX absolute action magnitude
    percentile: float
    source_files: tuple[str, ...]

    def clip_for(self, phase_id: int) -> np.ndarray:
        phase_id = int(phase_id)
        if phase_id < 0 or phase_id >= self.abs_clip.shape[0]:
            raise ValueError(f"invalid phase_id={phase_id}")
        return self.abs_clip[phase_id]


def _iter_place_npz(npz_dir: str | Path, seeds: Iterable[int] | None = None):
    root = Path(npz_dir)
    if seeds is None:
        files = sorted(root.glob("seed_*.npz"))
    else:
        files = [root / f"seed_{int(s)}.npz" for s in seeds]
    for path in files:
        if path.is_file():
            yield path


def load_expert_action_profile(
    npz_dir: str | Path,
    *,
    seeds: Iterable[int] | None = None,
    percentile: float = 95.0,
    min_clip: float = 5e-4,
    hard_cap: float = 2e-2,
) -> ExpertActionProfile:
    """Estimate per-stage action limits from successful PLACE demonstrations.

    ``percentile`` is applied independently to |dx|, |dy|, |dz| and |dyaw|.
    A small floor avoids a completely frozen controller when a dimension is very
    sparse, while ``hard_cap`` is only a final safety guard.  The normal case is
    that the measured demonstration percentile is substantially below it.
    """

    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be in (0, 100]")
    if min_clip <= 0 or hard_cap <= 0 or min_clip > hard_cap:
        raise ValueError("invalid min_clip/hard_cap")

    by_phase: list[list[np.ndarray]] = [[] for _ in range(NUM_PHASES)]
    source_files: list[str] = []
    for path in _iter_place_npz(npz_dir, seeds=seeds):
        d = np.load(path)
        if "cont_action" not in d or "micro_stage" not in d:
            continue
        cont = np.asarray(d["cont_action"], dtype=np.float32)
        phase = np.asarray(d["micro_stage"], dtype=np.int64)
        if cont.ndim != 2 or cont.shape[1] != 4 or len(cont) != len(phase):
            raise ValueError(f"bad M16 PLACE dataset shape in {path}")
        source_files.append(str(path))
        for ph in range(NUM_PHASES):
            rows = cont[phase == ph]
            if len(rows):
                by_phase[ph].append(rows)

    if not source_files:
        raise FileNotFoundError(
            f"no M16 PLACE seed_*.npz with cont_action/micro_stage found in {npz_dir}"
        )

    # Global fallback still comes from the demonstrations, never a guessed 0.05.
    all_rows = [x for groups in by_phase for x in groups]
    global_cont = np.concatenate(all_rows, axis=0)
    global_clip = np.percentile(np.abs(global_cont), percentile, axis=0)
    global_clip = np.clip(global_clip, min_clip, hard_cap).astype(np.float32)

    clips = np.zeros((NUM_PHASES, 4), dtype=np.float32)
    for ph in range(NUM_PHASES):
        if by_phase[ph]:
            rows = np.concatenate(by_phase[ph], axis=0)
            ph_clip = np.percentile(np.abs(rows), percentile, axis=0)
            clips[ph] = np.clip(ph_clip, min_clip, hard_cap)
        else:
            clips[ph] = global_clip

    # FAIL is terminal; keep a tiny no-op-compatible clip profile.
    clips[FAIL] = np.minimum(clips[FAIL], np.full(4, min_clip, np.float32))
    return ExpertActionProfile(
        abs_clip=clips.astype(np.float32),
        percentile=float(percentile),
        source_files=tuple(source_files),
    )


def phase_grip_command(phase_id: int) -> float:
    """Causal gripper label: only RELEASE is OPEN."""

    return 1.0 if int(phase_id) == RELEASE else -1.0


def phase_conditioned_expert_action(
    red_pos: np.ndarray,
    blue_pos: np.ndarray,
    phase_id: int,
    profile: ExpertActionProfile,
    *,
    stack_height_diff_m: float = DEFAULT_STACK_HEIGHT_DIFF_M,
) -> np.ndarray:
    """Return a 7-D DAgger expert action for the *current causal phase*.

    Motion behavior is deliberately conservative:
      ALIGN   : XY correction only, preserve height.
      DESCEND : XY correction + Z correction toward stack height.
      SETTLE  : smaller XYZ micro-correction while keeping the gripper closed.
      RELEASE : hold motion and OPEN.  No simultaneous XYZ command is taught.
      FAIL    : no-op CLOSE; the collector should stop before using it.
    """

    red = np.asarray(red_pos, dtype=np.float32).reshape(3)
    blue = np.asarray(blue_pos, dtype=np.float32).reshape(3)
    phase_id = int(phase_id)
    target = blue + np.asarray([0.0, 0.0, stack_height_diff_m], dtype=np.float32)
    err = target - red

    cont = np.zeros(4, dtype=np.float32)
    if phase_id == ALIGN:
        cont[:2] = err[:2]
    elif phase_id == DESCEND:
        cont[:3] = err[:3]
    elif phase_id == SETTLE:
        # terminal micro-correction; halve the geometric request before clipping
        # so DAgger does not inject a high-gain controller near contact.
        cont[:3] = 0.5 * err[:3]
    elif phase_id == RELEASE:
        cont[:] = 0.0
    elif phase_id == FAIL:
        cont[:] = 0.0
    else:
        raise ValueError(f"invalid phase_id={phase_id}")

    clip = profile.clip_for(phase_id)
    cont = np.clip(cont, -clip, clip).astype(np.float32)
    grip = phase_grip_command(phase_id)
    return np.asarray([cont[0], cont[1], cont[2], 0.0, 0.0, cont[3], grip], dtype=np.float32)


def summarize_action_profile(profile: ExpertActionProfile) -> dict[str, list[float]]:
    return {
        MICRO_NAMES[ph]: [float(x) for x in profile.abs_clip[ph]]
        for ph in range(NUM_PHASES)
    }
