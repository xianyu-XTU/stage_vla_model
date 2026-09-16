"""Physics descriptions independent of Isaac Lab classes."""

from .contacts import FRANKA_FINGER_CONTACTS, ContactSensorBinding
from .fixed_tilt import (
    FixedTiltReference,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    summarize_tilt_trace,
    wrap_angle,
)
from .grasp_geometry import (
    GraspGeometryProfile,
    ParallelJawCandidate,
    build_parallel_jaw_candidate,
    grasp_target_position,
    local_width_ratio,
    parallel_jaw_yaw_error,
    profile_for_geometry,
    profile_from_config,
)
from .limits import ControlLimits, RigidBodyLimits
from .materials import PhysicsMaterial
from .known_size import (
    grasp_contact_error,
    pressure_feedback_step,
    pressure_target_from_grip,
    pressure_tracking_ok,
    quaternion_control_angular_speed,
    size_conditioned_grasp_targets,
    stability_speed_from_source,
)
from .physical_grasp import (
    PhysicalGraspConfig,
    PhysicalGraspDiagnostics,
    physical_grasp,
    physical_grasp_diagnostics,
)
from .object_profiles import (
    ACTION_CONTEXT_ORDER,
    ACTION_CONTEXT_VERSION,
    GeometryAdapter,
    LoadAdapter,
    PhysicalDomainConfig,
    PhysicalObjectBatch,
    snapshot_has_physical_context,
)

__all__ = [
    "ACTION_CONTEXT_ORDER",
    "ACTION_CONTEXT_VERSION",
    "ContactSensorBinding",
    "ControlLimits",
    "FRANKA_FINGER_CONTACTS",
    "FixedTiltReference",
    "GraspGeometryProfile",
    "GeometryAdapter",
    "LoadAdapter",
    "ParallelJawCandidate",
    "PhysicalGraspConfig",
    "PhysicalGraspDiagnostics",
    "PhysicalDomainConfig",
    "PhysicalObjectBatch",
    "PhysicsMaterial",
    "RigidBodyLimits",
    "build_parallel_jaw_candidate",
    "euler_xyz_from_quat",
    "grasp_target_position",
    "grasp_contact_error",
    "local_width_ratio",
    "parallel_jaw_yaw_error",
    "physical_grasp",
    "physical_grasp_diagnostics",
    "pressure_feedback_step",
    "pressure_target_from_grip",
    "pressure_tracking_ok",
    "profile_for_geometry",
    "profile_from_config",
    "quat_from_euler_xyz",
    "quaternion_control_angular_speed",
    "size_conditioned_grasp_targets",
    "snapshot_has_physical_context",
    "stability_speed_from_source",
    "summarize_tilt_trace",
    "wrap_angle",
]
