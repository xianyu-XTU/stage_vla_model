"""M14 Semantic Action Tokenizer.

Represents one expert action step as:

    Stage Token + Continuous Action Token

The raw 7-D continuous action ``[dx, dy, dz, rx, ry, rz, grip]``
(pos delta in meters, axis-angle rotation delta in radians, gripper command)
is carried losslessly as float32; the stage label is mapped to a discrete
semantic token id. Encode/decode is a lossless bijection -- no discretization,
so the fine-grained continuous control information the grasp task depends on
(M9b/M10 DSL replay failure lesson) is never discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# stage vocabulary
# ---------------------------------------------------------------------------

STAGE_NAMES: tuple[str, ...] = ("REACH", "GRASP", "LIFT", "TRANSPORT", "PLACE")
"""Ordered stage vocabulary. Index == semantic token id."""

STAGE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(STAGE_NAMES)}
ID_TO_STAGE: dict[int, str] = {i: name for i, name in enumerate(STAGE_NAMES)}

NUM_STAGES = len(STAGE_NAMES)

# ---------------------------------------------------------------------------
# action layout
# ---------------------------------------------------------------------------

ACTION_DIM = 7  # dx,dy,dz, rx,ry,rz, grip
POS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 6)
GRIP_INDEX = 6
# combined Student target vector layout: [stage_id, dx,dy,dz, rx,ry,rz, grip]
VECTOR_DIM = ACTION_DIM + 1


@dataclass
class SemanticActionToken:
    """Structured, lossless representation of one expert action step."""

    stage_id: int
    delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    delta_rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    gripper: float = 0.0

    def to_vector(self) -> np.ndarray:
        """[stage_id, dx, dy, dz, rx, ry, rz, grip] -- Student action-head target."""
        return np.concatenate(
            [[self.stage_id], self.delta_xyz, self.delta_rotation, [self.gripper]]
        ).astype(np.float32, copy=False)

    @classmethod
    def from_vector(cls, vec) -> "SemanticActionToken":
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != VECTOR_DIM:
            raise ValueError(f"expected {VECTOR_DIM} elements, got {v.shape[0]}")
        return cls(
            stage_id=int(round(float(v[0]))),
            delta_xyz=v[1:4].copy(),
            delta_rotation=v[4:7].copy(),
            gripper=float(v[7]),
        )


class SemanticActionTokenizer:
    """Lossless encode/decode between 7-D actions and semantic tokens.

    ``encode``/``decode`` are exact float32 round-trips; ``reconstruction_error``
    reports how faithfully a decoded action reproduces the original.
    """

    def __init__(self, dtype: np.dtype = np.float32):
        self.dtype = np.dtype(dtype)

    # -- stage mapping ------------------------------------------------------

    def stage_to_id(self, stage: str) -> int:
        if stage not in STAGE_TO_ID:
            raise ValueError(f"unknown stage {stage!r}; known: {STAGE_NAMES}")
        return STAGE_TO_ID[stage]

    def id_to_stage(self, stage_id: int) -> str:
        return ID_TO_STAGE[int(stage_id)]

    # -- encode / decode ----------------------------------------------------

    def encode(self, action, stage: str) -> SemanticActionToken:
        a = np.asarray(action, dtype=self.dtype).reshape(-1)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}-D action, got {a.shape[0]}")
        return SemanticActionToken(
            stage_id=self.stage_to_id(stage),
            delta_xyz=a[POS_SLICE].astype(self.dtype, copy=True),
            delta_rotation=a[ROT_SLICE].astype(self.dtype, copy=True),
            gripper=float(a[GRIP_INDEX]),
        )

    def decode(self, token: SemanticActionToken) -> np.ndarray:
        a = np.empty(ACTION_DIM, dtype=self.dtype)
        a[POS_SLICE] = token.delta_xyz
        a[ROT_SLICE] = token.delta_rotation
        a[GRIP_INDEX] = token.gripper
        return a

    # -- vectorized form (Student target) -----------------------------------

    def token_to_vector(self, token: SemanticActionToken) -> np.ndarray:
        return token.to_vector()

    def vector_to_token(self, vec) -> SemanticActionToken:
        return SemanticActionToken.from_vector(vec)

    # -- reconstruction check ------------------------------------------------

    def reconstruction_error(self, original, decoded) -> dict[str, float]:
        """Max absolute error per group between original and decoded actions.

        Accepts a single 7-D action or a batch (N, 7). A lossless tokenizer
        yields ``max_abs_error`` == 0.0 (or float32 rounding ~1e-8 if the
        original was float64).
        """
        o = np.asarray(original, dtype=float)
        d = np.asarray(decoded, dtype=float)
        if o.shape != d.shape or o.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"original/decoded must share shape (..., {ACTION_DIM}); "
                f"got {o.shape} vs {d.shape}"
            )
        diff = np.abs(o - d)
        return {
            "pos_error": float(diff[..., POS_SLICE].max()),
            "rot_error": float(diff[..., ROT_SLICE].max()),
            "grip_error": float(diff[..., GRIP_INDEX].max()),
            "max_abs_error": float(diff.max()),
        }
