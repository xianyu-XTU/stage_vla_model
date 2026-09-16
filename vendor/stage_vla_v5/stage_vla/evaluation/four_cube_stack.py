"""Pure planning and layout utilities for the four-cube action audit.

This module deliberately does not launch Isaac Sim.  It owns the task plan,
safe layout generation, and evaluator argument contract so that CLI wrappers
remain thin and the non-simulator behavior can be unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence


FOUR_CUBE_COMMAND = (
    "STACK_CHAIN(red_cube>blue_cube,blue_cube>green_cube,green_cube>yellow_cube)"
)
LABEL_TO_ASSET = {
    "red_cube": "cube_2",
    "blue_cube": "cube_1",
    "green_cube": "cube_3",
    "yellow_cube": "cube_4",
}
FOUR_CUBE_ASSETS = tuple(sorted(LABEL_TO_ASSET.values()))
DEFAULT_FOUR_CUBE_LAYOUT = {
    "cube_1": (0.42, 0.12, 0.0203),
    "cube_2": (0.42, -0.12, 0.0203),
    "cube_3": (0.58, 0.12, 0.0203),
    "cube_4": (0.58, 0.0, 0.0203),
}


@dataclass(frozen=True)
class LayoutBounds:
    """Reachable table region used by randomized development layouts."""

    x_min_m: float = 0.40
    x_max_m: float = 0.60
    y_min_m: float = -0.14
    y_max_m: float = 0.14
    z_m: float = 0.0203
    minimum_separation_m: float = 0.09

    def validate(self) -> None:
        values = (
            self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m,
            self.z_m, self.minimum_separation_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("layout bounds must be finite")
        if self.x_min_m >= self.x_max_m or self.y_min_m >= self.y_max_m:
            raise ValueError("layout bounds must have positive area")
        if self.z_m <= 0.0 or self.minimum_separation_m <= 0.0:
            raise ValueError("layout height and separation must be positive")


@dataclass(frozen=True)
class LayoutBatch:
    """One XYZ row per environment for every scene asset."""

    mode: str
    positions: Mapping[str, tuple[tuple[float, float, float], ...]]
    seed: int | None = None
    minimum_separation_m: float | None = None

    @property
    def num_envs(self) -> int:
        lengths = {len(rows) for rows in self.positions.values()}
        if len(lengths) != 1:
            raise ValueError("all assets must contain the same number of layout rows")
        return next(iter(lengths), 0)

    def validate(self, *, required_assets: Sequence[str] = FOUR_CUBE_ASSETS) -> None:
        required = tuple(required_assets)
        if set(self.positions) != set(required):
            raise ValueError(
                f"layout assets must be exactly {sorted(required)}, got {sorted(self.positions)}"
            )
        if self.num_envs < 1:
            raise ValueError("layout batch must contain at least one environment")
        for asset, rows in self.positions.items():
            for row in rows:
                if len(row) != 3 or not all(math.isfinite(float(value)) for value in row):
                    raise ValueError(f"{asset} positions must contain finite XYZ triples")
        if self.minimum_separation_m is not None:
            if self.minimum_separation_m <= 0.0:
                raise ValueError("minimum separation must be positive")
            for env_index in range(self.num_envs):
                points = [self.positions[name][env_index] for name in required]
                for left in range(len(points)):
                    for right in range(left + 1, len(points)):
                        if math.dist(points[left][:2], points[right][:2]) + 1e-12 < self.minimum_separation_m:
                            raise ValueError(
                                f"environment {env_index} violates minimum XY separation"
                            )

    def as_manifest(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": "stage_vla_v5.asset_layout_batch.v1",
            "mode": self.mode,
            "seed": self.seed,
            "num_envs": self.num_envs,
            "minimum_separation_m": self.minimum_separation_m,
            "asset_positions_local_xyz": {
                name: [list(row) for row in rows]
                for name, rows in sorted(self.positions.items())
            },
        }


def _batch_from_env_layouts(
    mode: str,
    env_layouts: Sequence[Mapping[str, Sequence[float]]],
    *,
    seed: int | None,
    minimum_separation_m: float | None,
) -> LayoutBatch:
    positions = {
        name: tuple(
            tuple(float(value) for value in layout[name])  # type: ignore[misc]
            for layout in env_layouts
        )
        for name in FOUR_CUBE_ASSETS
    }
    batch = LayoutBatch(
        mode=mode,
        positions=positions,
        seed=seed,
        minimum_separation_m=minimum_separation_m,
    )
    batch.validate()
    return batch


def fixed_four_cube_layouts(num_envs: int) -> LayoutBatch:
    """Repeat the canonical layout across a parallel Isaac environment batch."""
    if int(num_envs) < 1:
        raise ValueError("num_envs must be positive")
    return _batch_from_env_layouts(
        "fixed",
        [DEFAULT_FOUR_CUBE_LAYOUT for _ in range(int(num_envs))],
        seed=None,
        minimum_separation_m=None,
    )


def sample_safe_four_cube_layouts(
    num_envs: int,
    *,
    seed: int,
    bounds: LayoutBounds = LayoutBounds(),
    max_attempts_per_environment: int = 20_000,
) -> LayoutBatch:
    """Sample reproducible reachable layouts with collision-free initial XY poses."""
    if int(num_envs) < 1:
        raise ValueError("num_envs must be positive")
    if int(max_attempts_per_environment) < 1:
        raise ValueError("max_attempts_per_environment must be positive")
    bounds.validate()
    rng = random.Random(int(seed))
    env_layouts: list[dict[str, tuple[float, float, float]]] = []
    for env_index in range(int(num_envs)):
        for _attempt in range(int(max_attempts_per_environment)):
            points = [
                (
                    rng.uniform(bounds.x_min_m, bounds.x_max_m),
                    rng.uniform(bounds.y_min_m, bounds.y_max_m),
                    bounds.z_m,
                )
                for _ in FOUR_CUBE_ASSETS
            ]
            separated = all(
                math.dist(points[left][:2], points[right][:2])
                >= bounds.minimum_separation_m
                for left in range(len(points))
                for right in range(left + 1, len(points))
            )
            if separated:
                env_layouts.append(dict(zip(FOUR_CUBE_ASSETS, points, strict=True)))
                break
        else:
            raise RuntimeError(
                f"could not sample safe four-cube layout for environment {env_index}"
            )
    return _batch_from_env_layouts(
        "random_safe",
        env_layouts,
        seed=int(seed),
        minimum_separation_m=bounds.minimum_separation_m,
    )


def write_layout_manifest(path: Path, layout: LayoutBatch) -> Path:
    """Persist one auditable layout batch without simulator dependencies."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(layout.as_manifest(), indent=2) + "\n", encoding="utf-8")
    return output


def load_layout_manifest(
    path: Path,
    *,
    required_assets: Sequence[str],
    expected_num_envs: int,
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Load and validate the evaluator-facing asset-position mapping."""
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage_vla_v5.asset_layout_batch.v1":
        raise ValueError(f"unsupported asset layout schema: {payload.get('schema')!r}")
    raw_positions = payload.get("asset_positions_local_xyz")
    if not isinstance(raw_positions, dict):
        raise ValueError("asset layout must contain asset_positions_local_xyz")
    try:
        positions = {
            str(name): tuple(tuple(float(value) for value in row) for row in rows)
            for name, rows in raw_positions.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("asset layout positions must be arrays of XYZ rows") from exc
    batch = LayoutBatch(
        mode=str(payload.get("mode", "external")),
        positions=positions,
        seed=payload.get("seed"),
        minimum_separation_m=payload.get("minimum_separation_m"),
    )
    batch.validate(required_assets=required_assets)
    if batch.num_envs != int(expected_num_envs):
        raise ValueError(
            f"asset layout has {batch.num_envs} environments, expected {expected_num_envs}"
        )
    return dict(batch.positions)


def compile_four_cube_task(root: Path) -> dict[str, object]:
    """Compile the DSL task and verify action-domain coverage before Isaac starts."""
    from stage_vla.action_runtime import generalized_cube_action_model_manifest
    from stage_vla.objects import ObjectModule
    from stage_vla.task_dsl import TaskCompiler
    from stage_vla.task_dsl.feasibility import assess_chain_capability

    compiler = TaskCompiler()
    chain = compiler.parse_chain(FOUR_CUBE_COMMAND)
    tokens = compiler.compile_chain(chain)
    relations = tuple(
        (task.object_name, task.target_name) for task in chain.execution_tasks
    )
    manifest = generalized_cube_action_model_manifest(Path(root))
    modules = tuple(ObjectModule(*relation) for relation in relations)
    unsupported = [module.labels() for module in modules if not manifest.supports_module(module)]
    capability = assess_chain_capability(tokens, object_domain=manifest.object_domain)
    if len(tokens) != 24 or unsupported or not capability.action_models_cover_all_stages:
        raise RuntimeError(
            f"four-cube task failed preflight: tokens={len(tokens)}, unsupported={unsupported}"
        )
    return {
        "command": FOUR_CUBE_COMMAND,
        "execution_relations": [list(relation) for relation in relations],
        "task_pairs": [
            f"{LABEL_TO_ASSET[object_name]}:{LABEL_TO_ASSET[support_name]}"
            for object_name, support_name in relations
        ],
        "skill_count": len(tokens),
        "action_model_domain": "cube/box/parallel_jaw; color-independent",
    }


def build_four_cube_eval_args(
    *,
    root: Path,
    output: Path,
    reach_checkpoint: Path,
    layout_manifest: Path,
    plan: Mapping[str, object],
    num_envs: int,
    seed: int,
    device: str,
    reach_recovery_steps: int = 0,
    final_stack_settle_steps: int = 40,
    dagger_dir: Path | None = None,
    reach_model_type: str = "independent_bc_policy",
) -> list[str]:
    """Build the stable full-chain evaluator contract for one Isaac process."""
    root = Path(root).resolve()
    bundle = root / "outputs/v5_generalized_cube_bc_v1"
    task_pairs = plan.get("task_pairs")
    if not isinstance(task_pairs, list) or not all(isinstance(value, str) for value in task_pairs):
        raise ValueError("plan task_pairs must be a list of strings")
    checkpoints = {
        "grasp": bundle / "grasp/policy.ts",
        "lift": bundle / "lift/policy.ts",
        "transport": bundle / "transport/policy.ts",
        "align": bundle / "align/policy.ts",
        "descend": bundle / "descend/policy.ts",
        "release": bundle / "release_stabilize/policy.ts",
        "retreat": bundle / "retreat/policy.ts",
    }
    required_files = [Path(reach_checkpoint), Path(layout_manifest), *checkpoints.values()]
    missing = [str(path) for path in required_files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing evaluation inputs: {missing}")
    if min(int(num_envs), 1 + int(reach_recovery_steps), 1 + int(final_stack_settle_steps)) < 1:
        raise ValueError("num_envs must be positive and recovery/settle steps non-negative")
    arguments = [
        "eval_v5_full_chain.py",
        "--config", str(root / "config/v5_generalized_cube_eval.json"),
        "--reach_checkpoint", str(Path(reach_checkpoint).resolve()),
        "--reach_model_type", str(reach_model_type),
        "--grasp_checkpoint", str(checkpoints["grasp"]),
        "--lift_checkpoint", str(checkpoints["lift"]),
        "--transport_checkpoint", str(checkpoints["transport"]),
        "--align_checkpoint", str(checkpoints["align"]),
        "--descend_checkpoint", str(checkpoints["descend"]),
        "--release_checkpoint", str(checkpoints["release"]),
        "--retreat_checkpoint", str(checkpoints["retreat"]),
        "--output", str(Path(output).resolve()),
        "--num_envs", str(int(num_envs)),
        "--seed", str(int(seed)),
        "--device", str(device),
        "--scene_cube_count", "4",
        "--task_pairs", *task_pairs,
        "--asset_layout_file", str(Path(layout_manifest).resolve()),
        "--reach_steps", "600",
        "--reach_recovery_steps", str(int(reach_recovery_steps)),
        "--grasp_steps", "160",
        "--lift_steps", "500",
        "--transport_steps", "500",
        "--align_steps", "650",
        "--descend_steps", "400",
        "--release_steps", "300",
        "--retreat_steps", "500",
        "--grasp_to_lift_settle_steps", "10",
        "--lift_to_transport_settle_steps", "10",
        "--transport_to_align_settle_steps", "10",
        "--align_to_descend_settle_steps", "10",
        "--final_stack_settle_steps", str(int(final_stack_settle_steps)),
    ]
    if dagger_dir is not None:
        arguments.extend((
            "--dagger_dir", str(Path(dagger_dir).resolve()),
            "--dagger_label_skills", "REACH",
        ))
    return arguments
