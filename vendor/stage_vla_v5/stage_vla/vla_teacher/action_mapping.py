"""Explicit OpenVLA-to-Isaac action convention helpers for the v4.1 audit."""

from __future__ import annotations

import numpy as np

OPENVLA_ACTION_NAMES = ("dx", "dy", "dz", "roll", "pitch", "yaw", "gripper_open")
ISAAC_ACTION_NAMES = ("dx_raw", "dy_raw", "dz_raw", "dRx_raw", "dRy_raw", "dRz_raw", "gripper")

LIBERO_OSC_POSE_CONTROL_HZ = 20.0
LIBERO_OSC_POSE_TRANSLATION_LIMIT_M = 0.05
LIBERO_OSC_POSE_ROTATION_LIMIT_RAD = 0.5


def openvla_gripper_to_isaac(value: np.ndarray | float) -> np.ndarray:
    """Map OpenVLA's 0=close, 1=open convention to Isaac's -1=close, +1=open."""
    gripper = np.asarray(value, dtype=np.float32)
    if not np.isfinite(gripper).all():
        raise ValueError("OpenVLA gripper contains NaN/Inf")
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValueError("OpenVLA gripper must be in [0,1] before Isaac mapping")
    return 2.0 * gripper - 1.0


def libero_openvla_gripper_to_isaac(
    value: np.ndarray | float,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binarize a LIBERO OpenVLA gripper channel for Isaac Lab.

    The fine-tuned OpenVLA checkpoint exposes its training convention after
    unnormalization: 0 means closed and 1 means open.  LIBERO's simulator
    subsequently flips that convention because robosuite expects +1 for close,
    but Isaac Lab directly expects a positive value for open and a negative
    value for close.  Therefore no LIBERO-environment sign flip belongs here.
    """
    gripper = np.asarray(value, dtype=np.float32)
    if not np.isfinite(gripper).all():
        raise ValueError("LIBERO OpenVLA gripper contains NaN/Inf")
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValueError("LIBERO OpenVLA gripper must be in [0,1] before Isaac mapping")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"gripper threshold must be finite and in [0,1], got {threshold}")
    return np.where(gripper >= float(threshold), 1.0, -1.0).astype(np.float32)


def _six_axis_scale(value: float | np.ndarray, *, name: str) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.repeat(scale.reshape(1), 6)
    scale = scale.reshape(-1)
    if scale.shape != (6,) or not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise ValueError(f"{name} must be one finite nonzero scalar or 6-vector, got {scale}")
    return scale


def _coordinate_rotation(value: np.ndarray | None) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"source_to_isaac_rotation must be a finite 3x3 matrix, got {matrix.shape}")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-5) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1.0e-5
    ):
        raise ValueError("source_to_isaac_rotation must be a proper rotation matrix")
    return matrix


def yaw_rotation_matrix(degrees: float) -> np.ndarray:
    """Return a source-to-target Z-axis rotation matrix."""
    if not np.isfinite(degrees):
        raise ValueError(f"yaw degrees must be finite, got {degrees}")
    radians = np.deg2rad(float(degrees))
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def libero_openvla_to_isaac_ik_rel(
    action: np.ndarray,
    *,
    isaac_arm_scale: float | np.ndarray = 0.5,
    source_control_hz: float = LIBERO_OSC_POSE_CONTROL_HZ,
    target_control_hz: float = LIBERO_OSC_POSE_CONTROL_HZ,
    translation_limit_m: float = LIBERO_OSC_POSE_TRANSLATION_LIMIT_M,
    rotation_limit_rad: float = LIBERO_OSC_POSE_ROTATION_LIMIT_RAD,
    source_to_isaac_rotation: np.ndarray | None = None,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """Convert a LIBERO OpenVLA action to Isaac Lab Franka IK-relative input.

    LIBERO / robosuite OSC_POSE actions are normalized controller commands:
    translation channels map [-1, 1] to +/-0.05 m and axis-angle rotation
    channels map [-1, 1] to +/-0.5 rad.  Isaac Lab then multiplies all six
    IK-relative channels by ``isaac_arm_scale``.  Per-step deltas are also
    rescaled by source_hz / target_hz to preserve commanded velocity when the
    simulation control rates differ.

    Both controllers left-multiply an axis-angle delta onto the current pose.
    In the current fixed-base Franka scenes their world / root axes coincide,
    so the default coordinate rotation is identity.  A proper 3x3 rotation can
    be supplied for deployments whose robot roots do not coincide.
    """
    source = np.asarray(action, dtype=np.float32)
    if source.shape[-1] != 7:
        raise ValueError(f"LIBERO OpenVLA action must end in 7 values, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("LIBERO OpenVLA action contains NaN/Inf")

    rates = np.asarray([source_control_hz, target_control_hz], dtype=np.float64)
    if not np.isfinite(rates).all() or np.any(rates <= 0.0):
        raise ValueError(f"control rates must be finite and positive, got {rates}")
    limits = np.asarray([translation_limit_m, rotation_limit_rad], dtype=np.float64)
    if not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise ValueError(f"LIBERO OSC limits must be finite and positive, got {limits}")

    coordinate_rotation = _coordinate_rotation(source_to_isaac_rotation)
    isaac_scale = _six_axis_scale(isaac_arm_scale, name="isaac_arm_scale")
    time_scale = float(source_control_hz) / float(target_control_hz)

    translation_m = np.einsum(
        "ij,...j->...i", coordinate_rotation, source[..., :3] * float(translation_limit_m)
    )
    rotation_axis_angle = np.einsum(
        "ij,...j->...i", coordinate_rotation, source[..., 3:6] * float(rotation_limit_rad)
    )
    physical_delta = np.concatenate((translation_m, rotation_axis_angle), axis=-1) * time_scale

    mapped = np.empty_like(source, dtype=np.float32)
    mapped[..., :6] = physical_delta / isaac_scale
    mapped[..., 6] = libero_openvla_gripper_to_isaac(source[..., 6], threshold=gripper_threshold)
    return mapped


def metric_translation_to_isaac_raw(
    delta_xyz_m: np.ndarray,
    *,
    arm_scale: float | np.ndarray,
) -> np.ndarray:
    """Invert Isaac's processed_xyz = raw_xyz * arm_action.scale contract."""
    delta = np.asarray(delta_xyz_m, dtype=np.float32)
    if delta.shape[-1] != 3:
        raise ValueError(f"delta_xyz_m must end in 3 values, got {delta.shape}")
    scale = np.asarray(arm_scale, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.repeat(scale.reshape(1), 3)
    scale = scale.reshape(-1)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise ValueError(f"arm_scale must be one finite nonzero scalar or 3-vector, got {scale}")
    return delta / scale


def openvla_to_isaac_translation_gripper(
    action: np.ndarray,
    *,
    arm_scale: float | np.ndarray = 0.5,
) -> np.ndarray:
    """Map the convention-verified XYZ and gripper channels.

    Rotation is deliberately copied through unchanged. OpenVLA documents
    roll/pitch/yaw deltas while the Isaac controller consumes a rotation-vector
    delta. v4.1 therefore does not claim full 7-D deployability from this helper.
    """
    source = np.asarray(action, dtype=np.float32)
    if source.shape[-1] != 7:
        raise ValueError(f"OpenVLA action must end in 7 values, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("OpenVLA action contains NaN/Inf")
    mapped = source.copy()
    mapped[..., :3] = metric_translation_to_isaac_raw(source[..., :3], arm_scale=arm_scale)
    mapped[..., 6] = openvla_gripper_to_isaac(source[..., 6])
    return mapped
