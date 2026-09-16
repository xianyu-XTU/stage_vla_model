"""Reinforcement-learning components for stage_vla."""

from .baseline_reward import (
    M9ABaselineRewardConfig,
    current_success_reward,
    exact_red_on_blue_goal,
    gate_reward_by_bool,
    gated_goal_tracking_reward,
    gripper_closedness,
    gripper_coordination_reward,
    lift_progress_reward,
    pregrasp_pose_reward,
    postgrasp_blended_target_z,
    postgrasp_pose_reward,
    placement_ready_mask,
    release_coordination_reward,
    smooth_distance_reward,
)

__all__ = [
    "M9ABaselineRewardConfig",
    "current_success_reward",
    "exact_red_on_blue_goal",
    "gate_reward_by_bool",
    "gated_goal_tracking_reward",
    "gripper_closedness",
    "gripper_coordination_reward",
    "lift_progress_reward",
    "pregrasp_pose_reward",
    "postgrasp_blended_target_z",
    "postgrasp_pose_reward",
    "placement_ready_mask",
    "release_coordination_reward",
    "smooth_distance_reward",
]

from .action_adapter import M9A_POLICY_ACTION_DIM, M9A_RAW_ACTION_DIM, expand_m9a_policy_action

from .edge_alignment import (
    EdgeAlignmentConfig,
    EdgeAlignmentRuntime,
    edge_aligned_raw_action,
    edge_alignment_runtime,
    load_edge_alignment_cfg,
)

from .action_dsl import (
    M9B_CATEGORY_COUNTS,
    M9B_CLOSE_TOKEN,
    M9B_GRIPPER_CATEGORIES,
    M9B_KEEP_TOKEN,
    M9B_OPEN_TOKEN,
    M9B_POLICY_FACTORS,
    M9B_TRANSLATION_CATEGORIES,
    M9B_TRANSLATION_MAX_ABS_BIN,
    M9BDecodedAction,
    decode_m9b_tokens,
    translation_bin_to_raw,
    translation_token_to_bin,
    validate_m9b_tokens,
)

__all__ += [
    "M9B_CATEGORY_COUNTS",
    "M9B_CLOSE_TOKEN",
    "M9B_GRIPPER_CATEGORIES",
    "M9B_KEEP_TOKEN",
    "M9B_OPEN_TOKEN",
    "M9B_POLICY_FACTORS",
    "M9B_TRANSLATION_CATEGORIES",
    "M9B_TRANSLATION_MAX_ABS_BIN",
    "M9BDecodedAction",
    "decode_m9b_tokens",
    "translation_bin_to_raw",
    "translation_token_to_bin",
    "validate_m9b_tokens",
]

from .m10_release_credit import (
    M10ReleaseCreditConfig,
    M10ReleaseCreditDiagnostics,
    M10ReleaseCreditTracker,
)

__all__ += [
    "M10ReleaseCreditConfig",
    "M10ReleaseCreditDiagnostics",
    "M10ReleaseCreditTracker",
]

from .known_size_grasp import (
    KnownSizeGraspConfig,
    known_size_raw_action,
    pressure_feedback_step,
    pressure_target_from_grip,
    pressure_tracking_ok,
    size_conditioned_grasp_targets,
    stability_speed_from_source,
)

__all__ += [
    "KnownSizeGraspConfig",
    "known_size_raw_action",
    "pressure_feedback_step",
    "pressure_target_from_grip",
    "pressure_tracking_ok",
    "size_conditioned_grasp_targets",
    "stability_speed_from_source",
]

from .object_physics import (
    GeometryAdapter,
    LoadAdapter,
    PhysicalDomainConfig,
    PhysicalObjectBatch,
)

__all__ += [
    "GeometryAdapter",
    "LoadAdapter",
    "PhysicalDomainConfig",
    "PhysicalObjectBatch",
]

from .transport_handoff import (
    TransportHandoffConfig,
    transport_handoff_ready,
    transport_settle_reward_terms,
)

__all__ += [
    "TransportHandoffConfig",
    "transport_handoff_ready",
    "transport_settle_reward_terms",
]
