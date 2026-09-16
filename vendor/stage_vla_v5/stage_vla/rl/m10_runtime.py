"""RSL-RL runner bridge for M10 Stage-aware Action-DSL PPO."""

from __future__ import annotations

from pathlib import Path

from .m10_factor_local_credit import M10_F1_CREDIT_SEMANTICS
from .m9b_runtime import build_m9b_runner_cfg, m9b_runner_cfg_dict, validate_m9b_dsl_config

M10_F1_ALGORITHM_CLASS = "stage_vla.rl.m10_factor_local_ppo:M10FactorLocalGripPPO"


def build_m10_runner_cfg(*, config_file: Path, seed: int, max_iterations: int | None = None):
    """Reuse M9-B categorical actor and frozen PPO hyperparameters."""
    validate_m9b_dsl_config(config_file)
    cfg, info = build_m9b_runner_cfg(
        config_file=config_file,
        seed=seed,
        max_iterations=max_iterations,
    )
    cfg.experiment_name = "stage_vla_m10_stage_aware_dsl"
    cfg.run_name = ""
    return cfg, info


def m10_runner_cfg_dict(runner_cfg) -> dict:
    """Install M10-12-F1 custom PPO while preserving actor/critic/distribution cfg."""
    cfg = m9b_runner_cfg_dict(runner_cfg)
    cfg["algorithm"]["class_name"] = M10_F1_ALGORITHM_CLASS
    cfg["algorithm"]["factor_local_credit_semantics"] = M10_F1_CREDIT_SEMANTICS
    # The semantic marker is useful in saved configs/logs but not an __init__
    # argument accepted by PPO. The custom algorithm constructor is resolved by
    # OnPolicyRunner, so remove the marker before algorithm construction in a
    # project-local wrapper? No: construct_algorithm forwards all algorithm keys.
    # Therefore keep the marker only at top-level metadata instead.
    cfg["m10_factor_local_credit_semantics"] = cfg["algorithm"].pop("factor_local_credit_semantics")
    return cfg
