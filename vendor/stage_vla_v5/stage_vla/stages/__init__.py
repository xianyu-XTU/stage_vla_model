"""Stage-related pure building blocks."""

from .geometry import (
    SegmentProjection,
    absolute_height_error,
    euclidean_distance,
    fingertip_gap,
    height_aligned,
    midpoint,
    point_between_fingertips,
    segment_projection,
    xy_distance,
)
from .physical_grasp import (
    PhysicalGraspConfig,
    PhysicalGraspDiagnostics,
    physical_grasp,
    physical_grasp_diagnostics,
)

__all__ = [
    "SegmentProjection",
    "PhysicalGraspConfig",
    "PhysicalGraspDiagnostics",
    "absolute_height_error",
    "euclidean_distance",
    "fingertip_gap",
    "height_aligned",
    "midpoint",
    "point_between_fingertips",
    "segment_projection",
    "xy_distance",
    "physical_grasp",
    "physical_grasp_diagnostics",
]


from .stable_grasp import (
    StableGraspConfig,
    StableGraspDiagnostics,
    StableGraspTracker,
)

__all__ += [
    "StableGraspConfig",
    "StableGraspDiagnostics",
    "StableGraspTracker",
]


from .lift import (
    LiftConfig,
    LiftDiagnostics,
    lift_diagnostics,
    object_lifted,
)

__all__ += [
    "LiftConfig",
    "LiftDiagnostics",
    "lift_diagnostics",
    "object_lifted",
]


from .placement import (
    RedOnBlueConfig,
    RedOnBlueDiagnostics,
    red_on_blue_diagnostics,
    safe_transport_upper_center_z,
    vertical_surface_clearance_m,
)
from .task_history import TaskHistoryDiagnostics, TaskHistoryTracker
from .red_on_blue_success import (
    RedOnBlueSuccessConfig,
    RedOnBlueSuccessDiagnostics,
    RedOnBlueSuccessTracker,
    success_candidate,
)

__all__ += [
    "RedOnBlueConfig",
    "RedOnBlueDiagnostics",
    "red_on_blue_diagnostics",
    "safe_transport_upper_center_z",
    "vertical_surface_clearance_m",
    "TaskHistoryDiagnostics",
    "TaskHistoryTracker",
    "RedOnBlueSuccessConfig",
    "RedOnBlueSuccessDiagnostics",
    "RedOnBlueSuccessTracker",
    "success_candidate",
]

from .stage_progress import (
    ManipulationStage,
    STAGE_NAMES,
    StageProgressConfig,
    StageProgressDiagnostics,
    StageProgressTracker,
    stage_name,
)
from .stage_reward import (
    StageAwareRewardConfig,
    StageAwareRewardDiagnostics,
    StageAwareRewardTracker,
    StagePotentialConfig,
    StagePotentialInputs,
    normalized_progress_potential,
    stage_potential,
)

__all__ += [
    "ManipulationStage",
    "STAGE_NAMES",
    "StageProgressConfig",
    "StageProgressDiagnostics",
    "StageProgressTracker",
    "stage_name",
    "StageAwareRewardConfig",
    "StageAwareRewardDiagnostics",
    "StageAwareRewardTracker",
    "StagePotentialConfig",
    "StagePotentialInputs",
    "normalized_progress_potential",
    "stage_potential",
]

from .grasp_alignment import (
    EdgeAlignmentCommand,
    GraspAlignmentDiagnostics,
    edge_alignment_command,
    grasp_alignment_diagnostics,
    quat_wxyz_planar_axes,
    square_projected_width_m,
    undirected_angle_deg,
)

from .grasp_geometry import (
    GraspGeometryProfile,
    ParallelJawCandidate,
    build_parallel_jaw_candidate,
    grasp_target_position,
    local_width_ratio,
    profile_from_config,
    profile_for_geometry,
)
from .object_motion import RigidObjectMotionProfile, motion_profile_from_config

__all__ += [
    "EdgeAlignmentCommand",
    "GraspAlignmentDiagnostics",
    "edge_alignment_command",
    "grasp_alignment_diagnostics",
    "quat_wxyz_planar_axes",
    "square_projected_width_m",
    "undirected_angle_deg",
    "GraspGeometryProfile",
    "ParallelJawCandidate",
    "build_parallel_jaw_candidate",
    "grasp_target_position",
    "local_width_ratio",
    "profile_from_config",
    "profile_for_geometry",
    "RigidObjectMotionProfile",
    "motion_profile_from_config",
]
