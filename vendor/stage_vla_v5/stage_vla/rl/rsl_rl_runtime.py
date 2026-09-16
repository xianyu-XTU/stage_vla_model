"""RSL-RL 5.x runner configuration for the M9-A PPO baseline."""

from __future__ import annotations

import importlib.metadata as metadata
from dataclasses import dataclass
from pathlib import Path

from packaging import version

from stage_vla.core import read_float, read_int


@dataclass(frozen=True)
class RslRlRuntimeInfo:
    installed_version: str
    minimum_version: str


def require_rsl_rl_5() -> RslRlRuntimeInfo:
    installed = metadata.version("rsl-rl-lib")
    minimum = "5.0.0"
    if version.parse(installed) < version.parse(minimum):
        raise RuntimeError(
            f"M9-A expects rsl-rl-lib >= {minimum}; found {installed}. "
            "Do not silently downgrade the project runner configuration."
        )
    return RslRlRuntimeInfo(installed_version=installed, minimum_version=minimum)


def build_m9a_runner_cfg(*, config_file: Path, seed: int, max_iterations: int | None = None):
    """Construct the current Isaac Lab/RSL-RL 5.x actor/critic config."""
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
        handle_deprecated_rsl_rl_cfg,
    )

    info = require_rsl_rl_5()

    hidden = [
        read_int(config_file, "m9a_ppo", "hidden_dim_1"),
        read_int(config_file, "m9a_ppo", "hidden_dim_2"),
        read_int(config_file, "m9a_ppo", "hidden_dim_3"),
    ]
    actor = RslRlMLPModelCfg(
        hidden_dims=hidden,
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=read_float(config_file, "m9a_ppo", "init_std")
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=hidden,
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=read_float(config_file, "m9a_ppo", "entropy_coef"),
        num_learning_epochs=read_int(config_file, "m9a_ppo", "num_learning_epochs"),
        num_mini_batches=read_int(config_file, "m9a_ppo", "num_mini_batches"),
        learning_rate=read_float(config_file, "m9a_ppo", "learning_rate"),
        schedule="adaptive",
        gamma=read_float(config_file, "m9a_ppo", "gamma"),
        lam=read_float(config_file, "m9a_ppo", "lam"),
        desired_kl=read_float(config_file, "m9a_ppo", "desired_kl"),
        max_grad_norm=1.0,
    )
    cfg = RslRlOnPolicyRunnerCfg(
        seed=int(seed),
        device="cuda:0",
        num_steps_per_env=read_int(config_file, "m9a_ppo", "num_steps_per_env"),
        max_iterations=(
            int(max_iterations)
            if max_iterations is not None
            else read_int(config_file, "m9a_ppo", "max_iterations")
        ),
        empirical_normalization=False,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        clip_actions=read_float(config_file, "m9a_ppo", "clip_actions"),
        check_for_nan=True,
        save_interval=read_int(config_file, "m9a_ppo", "save_interval"),
        experiment_name="stage_vla_m9a_continuous",
        run_name="",
        logger="tensorboard",
        actor=actor,
        critic=critic,
        algorithm=algorithm,
    )
    # Keep the same compatibility step used by Isaac Lab's official train.py.
    cfg = handle_deprecated_rsl_rl_cfg(cfg, info.installed_version)
    return cfg, info
