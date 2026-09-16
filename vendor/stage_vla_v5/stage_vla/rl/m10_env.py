"""Build the M10 stage-aware Action-DSL PPO environment variant.

M10 deliberately reuses the frozen M9-A4/M9-B low-level stack:
- same scene / observations before the project stage augmentation;
- same 7-D pose-relative IK ActionManager;
- same per-finger contact sensors and terminations;
- same deterministic edge-aligned yaw controller in the RSL wrapper;
- same PPO common hyperparameters and Action DSL vocabulary.

The task-reward side changes from the A4 concurrent non-stage shaping set to one
stateful M8 stage-aware term plus the same generic action/joint regularizers.
Keeping the stage reward in a *single* ManagerTermBase instance guarantees that
previous-potential/history state is updated exactly once per environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stage_vla.core import read_float, read_int

from .m10_stage_core import M10_STAGE_COUNT
from .m9a_env import configure_m9a_baseline_env


@dataclass(frozen=True)
class M10EnvBuildSummary:
    policy_action_factors: int
    category_counts: tuple[int, int, int, int]
    raw_action_dim_expected: int
    arm_command_type: str
    arm_relative: bool
    stage_observation_dim: int
    base_policy_observation_dim: int | None
    reward_terms: tuple[str, ...]
    env_step_dt_s: float
    stage_reward_manager_weight: float
    stage_gamma: float
    stage_transition_bonus: float
    success_bonus: float


def validate_m10_config(config_file: Path) -> int:
    """Validate the architectural M10 stage-observation contract."""
    stage_dim = read_int(config_file, "m10", "stage_observation_dim")
    if stage_dim != M10_STAGE_COUNT:
        raise ValueError(
            f"m10.stage_observation_dim must be {M10_STAGE_COUNT}, got {stage_dim}. "
            "The project stage vocabulary is REACH/GRASP/LIFT/TRANSPORT/PLACE."
        )
    return stage_dim


def m10_stage_reward_params(config_file: Path) -> dict[str, int | float]:
    """Return primitive-only RewardTermCfg params for the stateful M10 term.

    Isaac Lab configclasses recursively process manager-term params.  The
    project therefore keeps custom immutable dataclasses *out* of cfg.params
    and reconstructs them inside M10StageAwareRewardTerm.
    """
    return {
        # 1 = DSL policy GRIP token (default), 0 = continuous PPO state-based.
        "use_policy_gripper_token": read_int(
            config_file, "m10", "use_policy_gripper_token"
        ),
        "radial_tolerance_m": read_float(config_file, "grasp", "radial_tolerance_m"),
        "grasp_height_tolerance_m": read_float(config_file, "grasp", "height_tolerance_m"),
        "contact_force_threshold_n": read_float(config_file, "grasp", "contact_force_threshold_n"),
        "endpoint_margin": read_float(config_file, "grasp", "endpoint_margin"),
        "required_stable_grasp_steps": read_int(
            config_file, "stable_grasp", "required_consecutive_steps"
        ),
        "minimum_object_lift_delta_m": read_float(
            config_file, "lift", "minimum_object_lift_delta_m"
        ),
        "xy_tolerance_m": read_float(config_file, "red_on_blue_success", "xy_tolerance_m"),
        "target_height_diff_m": read_float(
            config_file, "red_on_blue_success", "target_height_diff_m"
        ),
        "placement_height_tolerance_m": read_float(
            config_file, "red_on_blue_success", "height_tolerance_m"
        ),
        "max_red_linear_speed_mps": read_float(
            config_file, "red_on_blue_success", "max_red_linear_speed_mps"
        ),
        "max_red_angular_speed_radps": read_float(
            config_file, "red_on_blue_success", "max_red_angular_speed_radps"
        ),
        "max_blue_linear_speed_mps": read_float(
            config_file, "red_on_blue_success", "max_blue_linear_speed_mps"
        ),
        "max_blue_angular_speed_radps": read_float(
            config_file, "red_on_blue_success", "max_blue_angular_speed_radps"
        ),
        "gripper_open_tolerance_m": read_float(
            config_file, "red_on_blue_success", "gripper_open_tolerance_m"
        ),
        "required_settle_steps": read_int(
            config_file, "red_on_blue_success", "required_settle_steps"
        ),
        "cube_size_m": read_float(config_file, "task", "cube_size_m"),
        "stage_gamma": read_float(config_file, "stage_reward", "gamma"),
        "stage_transition_bonus": read_float(
            config_file, "stage_reward", "stage_transition_bonus"
        ),
        "success_bonus": read_float(config_file, "stage_reward", "success_bonus"),
        "release_closed_joint_pos_m": read_float(
            config_file, "m10", "release_closed_joint_pos_m"
        ),
        "release_openness_progress_bonus": read_float(
            config_file, "m10", "release_openness_progress_bonus"
        ),
        "release_event_bonus": read_float(config_file, "m10", "release_event_bonus"),
        "release_open_token_bonus": read_float(config_file, "m10", "release_open_token_bonus"),
        "settle_shaping_weight": read_float(config_file, "m10", "settle_shaping_weight"),
        "settle_speed_ratio_cap": read_float(config_file, "m10", "settle_speed_ratio_cap"),
    }


def m10_reward_term_cfg_params(
    config_file: Path, use_policy_gripper_token: int | None = None
) -> dict[str, dict[str, int | float]]:
    """Wrap M10 parameters under one manager-visible argument.

    Isaac Lab 3.0 statically validates a callable-class term against the
    signature of ``Class.__call__``.  The stateful class needs the full project
    configuration during ``__init__``, but RewardManager also forwards
    ``cfg.params`` on every call.  Exposing one explicit ``m10_params`` argument
    keeps that static contract simple while preserving primitive-only nested
    values for configclass compatibility.
    """
    flat = m10_stage_reward_params(config_file)
    if use_policy_gripper_token is not None:
        flat["use_policy_gripper_token"] = int(use_policy_gripper_token)
    _assert_primitive_params(flat)
    return {"m10_params": flat}


def _assert_primitive_params(params: dict[str, int | float]) -> None:
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"M10 RewardTermCfg param {name!r} must be primitive int/float, got {type(value).__name__}"
            )


def configure_m10_stage_aware_env(
    env_cfg, *, config_file: Path, continuous_release: bool = False
) -> M10EnvBuildSummary:
    """Mutate a parsed stack cfg into M10 while preserving the frozen B stack.

    ``continuous_release=True`` makes the stage-aware reward derive the release
    command from the actual gripper state instead of a DSL policy GRIP token, so
    it can run with the continuous M9A 4-D action wrapper.
    """
    from isaaclab.envs import mdp as generic_mdp
    from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

    # Reuse the entire frozen A4 configuration first.  This preserves the
    # verified pose-IK control semantics, contact sensors and relevant
    # terminations.  We then replace only task reward composition.
    base = configure_m9a_baseline_env(env_cfg, config_file=config_file)
    stage_dim = validate_m10_config(config_file)
    reward_term_params = m10_reward_term_cfg_params(
        config_file,
        use_policy_gripper_token=0 if continuous_release else None,
    )
    params = reward_term_params["m10_params"]

    # Import lazily: this module contains Isaac ManagerTermBase and must not be
    # pulled into portable test collection through stage_vla.rl.__init__.
    from .m10_stage_reward_term import M10StageAwareRewardTerm

    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    if step_dt <= 0.0:
        raise ValueError(f"Invalid environment control dt: {step_dt}")

    # RewardManager multiplies every term by (weight * env.step_dt).  M8 was
    # validated as a per-control-step reward, so 1/dt preserves exactly that
    # scale when moving the validated reward into RewardManager.
    stage_manager_weight = 1.0 / step_dt

    # Keep only the two generic A4 regularizers beside the stage-aware task
    # term.  Retaining A4's concurrent task shaping here would make M10
    # 'A4 + stage reward' and blur the intended stage-aware mechanism.
    rewards = {
        "stage_aware": RewTerm(
            func=M10StageAwareRewardTerm,
            weight=stage_manager_weight,
            params=reward_term_params,
        ),
        "action_rate": RewTerm(
            func=generic_mdp.action_rate_l2,
            weight=read_float(config_file, "m9a_reward", "action_rate_weight"),
        ),
        "joint_vel": RewTerm(
            func=generic_mdp.joint_vel_l2,
            weight=read_float(config_file, "m9a_reward", "joint_vel_weight"),
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }
    env_cfg.rewards = rewards

    return M10EnvBuildSummary(
        policy_action_factors=4,
        category_counts=(5, 5, 5, 3),
        raw_action_dim_expected=base.raw_action_dim_expected,
        arm_command_type=base.arm_command_type,
        arm_relative=base.arm_relative,
        stage_observation_dim=stage_dim,
        base_policy_observation_dim=None,
        reward_terms=tuple(rewards.keys()),
        env_step_dt_s=step_dt,
        stage_reward_manager_weight=stage_manager_weight,
        stage_gamma=float(params["stage_gamma"]),
        stage_transition_bonus=float(params["stage_transition_bonus"]),
        success_bonus=float(params["success_bonus"]),
    )
