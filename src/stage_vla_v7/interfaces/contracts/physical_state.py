"""Simulator-neutral contract for batched physical feedback."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from collections.abc import Iterator, Mapping


@dataclass(frozen=True)
class PhysicalState(Mapping[str, object]):
    """Batched physical tensors consumed by the frozen cube-policy runtime.

    Tensor ownership, shape, dtype, and device validation belong to the
    simulator adapter. Keeping these values opaque preserves the dependency-
    free interfaces layer while making the runtime state schema explicit.
    """

    object_position: object
    support_position: object
    object_linear_velocity: object
    object_angular_velocity: object
    object_orientation: object
    support_orientation: object
    gripper_open: object
    gripper_joint_position: object
    object_speed: object
    end_effector_position: object
    left_fingertip_position: object
    right_fingertip_position: object
    fingertip_force: object
    physical_grasp: object
    between_fingertips: object
    left_height_aligned: object
    right_height_aligned: object
    finger_a_contact: object
    finger_b_contact: object
    grasp_target: object
    radial_error: object
    left_height_error: object
    right_height_error: object
    projection_alpha: object
    fingertip_gap: object
    arm_joint_position: object
    arm_joint_velocity: object
    control_speed: object | None = None
    stability_speed: object | None = None
    control_angular_speed: object | None = None
    instantaneous_angular_speed: object | None = None
    stability_angular_speed: object | None = None

    _LEGACY_ALIASES = {
        "red": "object_position",
        "blue": "support_position",
        "vel": "object_linear_velocity",
        "angular": "object_angular_velocity",
        "red_quat": "object_orientation",
        "blue_quat": "support_orientation",
        "open": "gripper_open",
        "grip": "gripper_joint_position",
        "speed": "object_speed",
        "ee": "end_effector_position",
        "left_tip": "left_fingertip_position",
        "right_tip": "right_fingertip_position",
        "force": "fingertip_force",
        "physical": "physical_grasp",
        "grasp_target": "grasp_target",
        "radial_error_m": "radial_error",
        "left_height_error_m": "left_height_error",
        "right_height_error_m": "right_height_error",
        "fingertip_gap_m": "fingertip_gap",
        "q": "arm_joint_position",
        "qd": "arm_joint_velocity",
    }

    def __getitem__(self, key: str) -> object:
        name = self._LEGACY_ALIASES.get(key, key)
        if not hasattr(self, name) or name.startswith("_"):
            raise KeyError(key)
        value = getattr(self, name)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_legacy_mapping())

    def __len__(self) -> int:
        return len(self.to_legacy_mapping())

    @classmethod
    def from_legacy_mapping(cls, values: Mapping[str, object]) -> "PhysicalState":
        """Construct the contract from the historical measurement dictionary."""
        aliases = {value: key for key, value in cls._LEGACY_ALIASES.items()}
        payload: dict[str, object] = {}
        missing: list[str] = []
        for item in fields(cls):
            key = aliases.get(item.name, item.name)
            if key in values:
                payload[item.name] = values[key]
            elif item.default is None:
                payload[item.name] = None
            else:
                missing.append(key)
        if missing:
            raise ValueError(f"missing physical-state fields: {missing}")
        return cls(**payload)

    def with_motion_metrics(self, **values: object) -> "PhysicalState":
        allowed = {
            "control_speed",
            "stability_speed",
            "control_angular_speed",
            "instantaneous_angular_speed",
            "stability_angular_speed",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown motion metrics: {unknown}")
        return replace(self, **values)

    def to_legacy_mapping(self) -> dict[str, object]:
        """Expose the temporary dictionary ABI used by frozen helper code."""
        values = {
            "red": self.object_position,
            "blue": self.support_position,
            "vel": self.object_linear_velocity,
            "angular": self.object_angular_velocity,
            "red_quat": self.object_orientation,
            "blue_quat": self.support_orientation,
            "open": self.gripper_open,
            "grip": self.gripper_joint_position,
            "speed": self.object_speed,
            "ee": self.end_effector_position,
            "left_tip": self.left_fingertip_position,
            "right_tip": self.right_fingertip_position,
            "force": self.fingertip_force,
            "physical": self.physical_grasp,
            "between_fingertips": self.between_fingertips,
            "left_height_aligned": self.left_height_aligned,
            "right_height_aligned": self.right_height_aligned,
            "finger_a_contact": self.finger_a_contact,
            "finger_b_contact": self.finger_b_contact,
            "grasp_target": self.grasp_target,
            "radial_error_m": self.radial_error,
            "left_height_error_m": self.left_height_error,
            "right_height_error_m": self.right_height_error,
            "projection_alpha": self.projection_alpha,
            "fingertip_gap_m": self.fingertip_gap,
            "q": self.arm_joint_position,
            "qd": self.arm_joint_velocity,
        }
        for name in (
            "control_speed",
            "stability_speed",
            "control_angular_speed",
            "instantaneous_angular_speed",
            "stability_angular_speed",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return values
