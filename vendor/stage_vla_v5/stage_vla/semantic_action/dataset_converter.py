"""M14: convert M13 Teacher Dataset npz -> M14 Semantic Dataset npz.

Input  (outputs/m13_teacher_dataset/seed_XXXX.npz):
    images, actions, stages, instruction, seed

Output (outputs/m14_semantic_dataset/seed_XXXX.npz):
    images              (N, H, W, 3) uint8     -- unchanged
    stage_token         (N,) int64             -- discrete Stage Predictor target
    semantic_action_token (N, 8) float32       -- [stage_id, dx,dy,dz, rx,ry,rz, grip]
                                                   Semantic Action Head target
    continuous_action   (N, 7) float32         -- original 7-D raw action (ground truth)

Also reports per-seed reconstruction error (decode(semantic_action_token) vs
continuous_action). Lossless tokenizer => max_abs_error == 0.

Usage:
    python -m stage_vla.semantic_action.dataset_converter \
        --npz_dir outputs/m13_teacher_dataset \
        --out_dir  outputs/m14_semantic_dataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tokenizer import (
    ACTION_DIM,
    ID_TO_STAGE,
    SemanticActionTokenizer,
    VECTOR_DIM,
)


def convert_m13_to_m14(
    npz_dir: str | Path,
    out_dir: str | Path,
    only_seeds: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Convert every seed_*.npz in ``npz_dir`` to M14 semantic dataset.

    Returns per-seed max reconstruction error summaries.
    """
    npz_dir = Path(npz_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SemanticActionTokenizer(dtype=np.float32)

    if only_seeds:
        seeds = only_seeds
    else:
        seeds = sorted(
            int(p.stem.split("_")[1])
            for p in npz_dir.glob("seed_*.npz")
        )
    if not seeds:
        raise FileNotFoundError(f"no seed_*.npz found under {npz_dir}")

    summary: dict[int, dict[str, float]] = {}
    for seed in seeds:
        src = npz_dir / f"seed_{seed}.npz"
        if not src.is_file():
            print(f"[M14] skip missing {src}")
            continue
        data = np.load(src, allow_pickle=True)

        images = np.asarray(data["images"])          # (N,H,W,3) uint8
        actions = np.asarray(data["actions"], dtype=np.float32)  # (N,7)
        stages = np.asarray(data["stages"])          # (N,) str
        instruction = str(np.asarray(data["instruction"]).item())

        N = actions.shape[0]
        stage_token = np.zeros(N, dtype=np.int64)
        sem_vec = np.zeros((N, VECTOR_DIM), dtype=np.float32)
        decoded = np.zeros_like(actions)
        for i in range(N):
            stage = str(stages[i])
            tok = tokenizer.encode(actions[i], stage)
            stage_token[i] = tok.stage_id
            sem_vec[i] = tok.to_vector()
            decoded[i] = tokenizer.decode(tok)

        err = tokenizer.reconstruction_error(actions, decoded)
        summary[seed] = err

        out = out_dir / f"seed_{seed}.npz"
        np.savez_compressed(
            out,
            images=images,
            instruction=np.asarray(instruction),
            stage_token=stage_token,
            semantic_action_token=sem_vec,
            continuous_action=actions,
            seed=np.asarray(seed),
        )
        dist = dict(Counter(stage_token.tolist()))
        dist = {ID_TO_STAGE[k]: v for k, v in sorted(dist.items())}
        print(
            f"[M14] seed {seed}: {N} frames -> {out.name} "
            f"(img {images.shape[1]}x{images.shape[2]}, "
            f"pos_err {err['pos_error']:.2e} rot_err {err['rot_error']:.2e} "
            f"grip_err {err['grip_error']:.2e})"
        )
        print(f"      stage dist: {dist}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="M13 -> M14 semantic dataset converter")
    parser.add_argument("--npz_dir", type=str, default="outputs/m13_teacher_dataset")
    parser.add_argument("--out_dir", type=str, default="outputs/m14_semantic_dataset")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="only convert these seeds (default: all seed_*.npz)")
    args = parser.parse_args()

    convert_m13_to_m14(args.npz_dir, args.out_dir, only_seeds=args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
