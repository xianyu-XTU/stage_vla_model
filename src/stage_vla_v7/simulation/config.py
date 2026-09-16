"""Dependency-light configuration shared by Simulation backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class RigidObjectMotionProfile:
    """Maximum Cartesian translation per control step for each moving Skill."""

    lift_translation_limit_m: float = 0.005
    transport_translation_limit_m: float = 0.005
    align_translation_limit_m: float = 0.005
    descend_translation_limit_m: float = 0.003
    release_translation_limit_m: float = 0.003
    retreat_translation_limit_m: float = 0.005

    def validate(self) -> None:
        values = (
            self.lift_translation_limit_m,
            self.transport_translation_limit_m,
            self.align_translation_limit_m,
            self.descend_translation_limit_m,
            self.release_translation_limit_m,
            self.retreat_translation_limit_m,
        )
        if any(not 0.0 < float(value) <= 0.01 for value in values):
            raise ValueError("motion translation limits must lie in (0, 0.01]")

    def for_skill(self, skill: str) -> float:
        name = "release" if str(skill) == "RELEASE_STABILIZE" else str(skill).lower()
        key = f"{name}_translation_limit_m"
        if not hasattr(self, key):
            raise ValueError(f"skill has no motion limit: {skill!r}")
        return float(getattr(self, key))


def motion_profile_from_config(path: str | Path) -> RigidObjectMotionProfile:
    """Load an optional ``motion`` block while preserving frozen defaults."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("motion", {})
    defaults = RigidObjectMotionProfile()
    profile = RigidObjectMotionProfile(
        **{
            field: float(values.get(field, getattr(defaults, field)))
            for field in defaults.__dataclass_fields__
        }
    )
    profile.validate()
    return profile


@dataclass(frozen=True)
class KnownSizeGraspConfig:
    """Frozen physical assumptions for one known-size object class."""

    width_m: float
    depth_m: float
    height_m: float
    mass_kg: float
    grasp_width_m: float | None = None
    jaw_clearance_m: float = 0.002
    max_compression_m: float = 0.004
    friction_coefficient: float = 0.4
    safety_factor: float = 2.0
    min_force_n: float = 5.0
    max_force_n: float = 40.0
    pressure_tolerance_n: float = 1.0
    force_balance_tolerance_n: float = 1.5
    residual_force_range_n: float = 4.0
    residual_action_penalty: float = 0.5
    lift_acceleration_mps2: float = 0.5
    joint_min_m: float = 0.0
    joint_max_m: float = 0.04

    def validate(self) -> None:
        grasp_width = self.width_m if self.grasp_width_m is None else self.grasp_width_m
        positive = (
            self.width_m,
            self.depth_m,
            self.height_m,
            self.mass_kg,
            grasp_width,
            self.friction_coefficient,
            self.safety_factor,
            self.min_force_n,
            self.max_force_n,
            self.pressure_tolerance_n,
            self.force_balance_tolerance_n,
            self.joint_max_m,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("size, mass, friction, force and joint limits must be positive")
        nonnegative = (
            self.jaw_clearance_m,
            self.max_compression_m,
            self.lift_acceleration_mps2,
            self.joint_min_m,
            self.residual_force_range_n,
            self.residual_action_penalty,
        )
        if any(float(value) < 0 for value in nonnegative):
            raise ValueError(
                "clearance, compression, acceleration and joint minimum must be non-negative"
            )
        if self.min_force_n > self.max_force_n:
            raise ValueError("min_force_n must not exceed max_force_n")
        if self.joint_min_m >= self.joint_max_m:
            raise ValueError("joint_min_m must be less than joint_max_m")
        if grasp_width / 2.0 + self.jaw_clearance_m > self.joint_max_m:
            raise ValueError("object width plus clearance exceeds gripper opening")

    @property
    def geometric_joint_target_m(self) -> float:
        self.validate()
        grasp_width = self.width_m if self.grasp_width_m is None else self.grasp_width_m
        return min(self.joint_max_m, grasp_width / 2.0 + self.jaw_clearance_m)

    @property
    def compression_joint_min_m(self) -> float:
        self.validate()
        grasp_width = self.width_m if self.grasp_width_m is None else self.grasp_width_m
        return max(self.joint_min_m, grasp_width / 2.0 - self.max_compression_m)

    @property
    def initial_force_target_n(self) -> float:
        self.validate()
        required = (
            self.safety_factor
            * self.mass_kg
            * (9.81 + self.lift_acceleration_mps2)
            / (2.0 * self.friction_coefficient)
        )
        return min(self.max_force_n, max(self.min_force_n, required))

    def size_features(self, *, dtype=None, device=None):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional tensor helper
            raise RuntimeError("size features require torch") from exc
        if dtype is None:
            dtype = torch.float32
        self.validate()
        return torch.tensor(
            [self.width_m, self.depth_m, self.height_m],
            dtype=dtype,
            device=device,
        ) / 0.1


def load_known_size_config(path: str | Path) -> KnownSizeGraspConfig:
    """Load the frozen known-size/gripper contract from task JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    obj = payload["object"]
    grip = payload["gripper"]
    return KnownSizeGraspConfig(
        width_m=float(obj["width_m"]),
        depth_m=float(obj["depth_m"]),
        height_m=float(obj["height_m"]),
        mass_kg=float(obj["mass_kg"]),
        **{
            key: float(grip[key])
            for key in (
                "jaw_clearance_m",
                "max_compression_m",
                "friction_coefficient",
                "safety_factor",
                "min_force_n",
                "max_force_n",
                "pressure_tolerance_n",
                "force_balance_tolerance_n",
                "residual_force_range_n",
                "residual_action_penalty",
                "lift_acceleration_mps2",
                "joint_min_m",
                "joint_max_m",
            )
        },
    )


def load_object_metadata(path: str | Path) -> tuple[str, tuple[int, int, int]]:
    """Load the geometry label and RGB identity used by evaluation."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))["object"]
    return str(obj.get("geometry", "box")), tuple(
        int(value) for value in obj.get("color_rgb", (220, 40, 40))
    )


@dataclass(frozen=True)
class CameraSpec:
    """Backend-neutral camera declaration consumed by an environment factory."""

    name: str
    role: str
    width: int
    height: int
    data_types: tuple[str, ...]
    position_m: tuple[float, float, float] = (1.0, 0.0, 0.40)
    rotation_wxyz: tuple[float, float, float, float] = (
        -0.61237,
        -0.61237,
        0.35355,
        0.35355,
    )
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 20.955
    update_period_s: float = 0.05

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("camera name must be non-empty")
        if self.role not in {"vision", "observer"}:
            raise ValueError("camera role must be vision or observer")
        if min(int(self.width), int(self.height)) < 1:
            raise ValueError("camera dimensions must be positive")
        if not self.data_types or any(
            value not in {"rgb", "distance_to_image_plane"}
            for value in self.data_types
        ):
            raise ValueError("camera data types must contain supported outputs")
        if self.role == "vision" and not {
            "rgb",
            "distance_to_image_plane",
        }.issubset(self.data_types):
            raise ValueError("Vision camera must provide RGB and depth")
        if self.role == "observer" and self.data_types != ("rgb",):
            raise ValueError("Observer camera must provide RGB only")
        numeric = (
            *self.position_m,
            *self.rotation_wxyz,
            self.focal_length_mm,
            self.horizontal_aperture_mm,
            self.update_period_s,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("camera pose and optics must be finite")
        if min(
            float(self.focal_length_mm),
            float(self.horizontal_aperture_mm),
            float(self.update_period_s),
        ) <= 0.0:
            raise ValueError("camera optics and update period must be positive")


def validate_camera_specs(specs: tuple[CameraSpec, ...]) -> None:
    names = tuple(spec.name for spec in specs)
    roles = tuple(spec.role for spec in specs)
    if len(names) != len(set(names)):
        raise ValueError("camera IDs must be unique")
    if len(roles) != len(set(roles)):
        raise ValueError("only one camera may own each camera role")
