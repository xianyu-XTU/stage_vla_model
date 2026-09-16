"""RSL-RL runner/configuration bridge for M9-B factorized categorical PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stage_vla.core import read_int

from .action_dsl import (
    M9B_CATEGORY_COUNTS,
    M9B_CLOSE_TOKEN,
    M9B_GRIPPER_CATEGORIES,
    M9B_KEEP_TOKEN,
    M9B_OPEN_TOKEN,
    M9B_TRANSLATION_CATEGORIES,
    M9B_TRANSLATION_MAX_ABS_BIN,
)
from .rsl_rl_runtime import build_m9a_runner_cfg

M9B_DISTRIBUTION_CLASS = "stage_vla.rl.factorized_categorical:FactorizedCategoricalDistribution"


@dataclass(frozen=True)
class M9BDSLConfig:
    translation_categories: int
    translation_max_abs_bin: int
    gripper_categories: int
    open_token: int
    keep_token: int
    close_token: int


def validate_m9b_dsl_config(config_file: Path) -> M9BDSLConfig:
    """Read config metadata and reject drift from the frozen M9-B DSL API.

    The token vocabulary is an architectural interface implemented in code;
    ``default.yaml`` mirrors it so experiment artifacts explicitly record the
    semantics.  This function makes any accidental code/config divergence fail
    before training.
    """
    cfg = M9BDSLConfig(
        translation_categories=read_int(config_file, "m9b_dsl", "translation_categories"),
        translation_max_abs_bin=read_int(config_file, "m9b_dsl", "translation_max_abs_bin"),
        gripper_categories=read_int(config_file, "m9b_dsl", "gripper_categories"),
        open_token=read_int(config_file, "m9b_dsl", "open_token"),
        keep_token=read_int(config_file, "m9b_dsl", "keep_token"),
        close_token=read_int(config_file, "m9b_dsl", "close_token"),
    )
    expected = M9BDSLConfig(
        translation_categories=M9B_TRANSLATION_CATEGORIES,
        translation_max_abs_bin=M9B_TRANSLATION_MAX_ABS_BIN,
        gripper_categories=M9B_GRIPPER_CATEGORIES,
        open_token=M9B_OPEN_TOKEN,
        keep_token=M9B_KEEP_TOKEN,
        close_token=M9B_CLOSE_TOKEN,
    )
    if cfg != expected:
        raise ValueError(f"m9b_dsl config/code mismatch: config={cfg}, expected={expected}")
    return cfg


def build_m9b_runner_cfg(*, config_file: Path, seed: int, max_iterations: int | None = None):
    """Reuse frozen A4 PPO hyperparameters and change only actor action distribution."""
    validate_m9b_dsl_config(config_file)
    cfg, info = build_m9a_runner_cfg(config_file=config_file, seed=seed, max_iterations=max_iterations)
    cfg.experiment_name = "stage_vla_m9b_action_dsl"
    cfg.run_name = ""
    return cfg, info


def m9b_runner_cfg_dict(runner_cfg) -> dict:
    """Convert Isaac configclass to RSL-RL dict and install the project distribution.

    This late dictionary override avoids depending on Isaac Lab's configclass
    type annotation for custom distribution configs while following RSL-RL
    5.x's qualified ``resolve_callable`` extension path.
    """
    cfg = runner_cfg.to_dict()
    cfg["actor"]["distribution_cfg"] = {
        "class_name": M9B_DISTRIBUTION_CLASS,
        "category_counts": list(M9B_CATEGORY_COUNTS),
    }
    return cfg
