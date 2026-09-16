"""Canonical object catalog shared by task, vision and action adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


OBJECT_CATEGORIES = ("cube", "cup", "cloth", "generic")
GEOMETRY_TYPES = ("box", "cylinder", "cone", "capsule", "deformable_mesh", "unknown")
GRASP_MODES = ("parallel_jaw", "top_rim", "surface_clamp", "unknown")
SUPPORT_MODES = ("flat", "concave", "deformable", "unknown")

# Stable, label-free feature ABI for object-conditioned action models. Names
# and colors remain in the perception/task layer and never enter this vector.
ACTION_CONTEXT_VERSION = "stage_vla_v5.object_physics.v1"
ACTION_CONTEXT_ORDER = (
    "object_size_x", "object_size_y", "object_size_z",
    "support_size_x", "support_size_y", "support_size_z",
    "object_mass", "support_mass", "object_deformable", "stackable",
)


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    color_rgb: tuple[int, int, int] | None
    size_m: tuple[float, float, float] = (0.04, 0.04, 0.04)
    category: str = "generic"
    geometry: str = "unknown"
    mass_kg: float = 0.05
    grasp_mode: str = "unknown"
    support_mode: str = "unknown"
    deformable: bool = False
    stackable: bool = False
    friction_coefficient: float = 0.4

    def __post_init__(self) -> None:
        if not self.label or self.label.strip() != self.label:
            raise ValueError("label must be a non-empty trimmed string")
        if self.color_rgb is not None:
            if len(self.color_rgb) != 3 or any(not 0 <= int(value) <= 255 for value in self.color_rgb):
                raise ValueError("color_rgb must contain three values in [0,255]")
        if len(self.size_m) != 3 or any(float(value) <= 0 for value in self.size_m):
            raise ValueError("size_m must contain three positive values")
        if self.category not in OBJECT_CATEGORIES:
            raise ValueError(f"unsupported object category: {self.category!r}")
        if self.geometry not in GEOMETRY_TYPES:
            raise ValueError(f"unsupported geometry type: {self.geometry!r}")
        if self.grasp_mode not in GRASP_MODES:
            raise ValueError(f"unsupported grasp mode: {self.grasp_mode!r}")
        if self.support_mode not in SUPPORT_MODES:
            raise ValueError(f"unsupported support mode: {self.support_mode!r}")
        if float(self.mass_kg) <= 0:
            raise ValueError("mass_kg must be positive")
        if not math.isfinite(float(self.friction_coefficient)) or float(self.friction_coefficient) <= 0:
            raise ValueError("friction_coefficient must be finite and positive")

    @property
    def height_m(self) -> float:
        return float(self.size_m[2])

    @property
    def class_id(self) -> str:
        """Color-independent class used by action-model routing."""
        return self.category


@dataclass(frozen=True)
class ObjectModuleOutput:
    """Typed object-role output consumed by the action module.

    Colors remain available for diagnostics and visual association, while
    ``action_context`` deliberately contains only physical/class features.
    """

    object_label: str
    support_label: str
    object_class: str
    support_class: str
    object_geometry: str
    support_geometry: str
    object_size_m: tuple[float, float, float]
    support_size_m: tuple[float, float, float]
    object_mass_kg: float
    support_mass_kg: float
    object_grasp_mode: str
    support_mode: str
    object_deformable: bool
    stackable: bool
    object_friction_coefficient: float = 0.4
    support_friction_coefficient: float = 0.4
    object_color_rgb: tuple[int, int, int] | None = None
    support_color_rgb: tuple[int, int, int] | None = None

    def action_context(self) -> dict[str, object]:
        """Return the color-free features allowed across the action boundary."""
        return {
            "object_class": self.object_class,
            "support_class": self.support_class,
            "object_geometry": self.object_geometry,
            "support_geometry": self.support_geometry,
            "object_size_m": self.object_size_m,
            "support_size_m": self.support_size_m,
            "object_mass_kg": self.object_mass_kg,
            "support_mass_kg": self.support_mass_kg,
            "object_friction_coefficient": self.object_friction_coefficient,
            "support_friction_coefficient": self.support_friction_coefficient,
            "object_grasp_mode": self.object_grasp_mode,
            "support_mode": self.support_mode,
            "object_deformable": self.object_deformable,
            "stackable": self.stackable,
        }


@dataclass(frozen=True)
class ObjectDomain:
    """Color-independent physical domain covered by an action-model bundle."""

    object_classes: tuple[str, ...]
    support_classes: tuple[str, ...]
    object_geometries: tuple[str, ...]
    support_geometries: tuple[str, ...]
    object_grasp_modes: tuple[str, ...]
    support_modes: tuple[str, ...]
    object_size_min_m: tuple[float, float, float]
    object_size_max_m: tuple[float, float, float]
    support_size_min_m: tuple[float, float, float]
    support_size_max_m: tuple[float, float, float]
    object_mass_range_kg: tuple[float, float]
    allow_deformable: bool = False
    support_mass_range_kg: tuple[float, float] | None = None

    def validate(self) -> None:
        groups = (
            self.object_classes,
            self.support_classes,
            self.object_geometries,
            self.support_geometries,
            self.object_grasp_modes,
            self.support_modes,
        )
        if any(not values or len(set(values)) != len(values) for values in groups):
            raise ValueError("object-domain capability groups must be non-empty and unique")
        ranges = (
            (self.object_size_min_m, self.object_size_max_m),
            (self.support_size_min_m, self.support_size_max_m),
        )
        for lower, upper in ranges:
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError("object-domain size ranges must be xyz triples")
            if any(float(low) <= 0 or float(low) > float(high) for low, high in zip(lower, upper)):
                raise ValueError("object-domain size ranges must be positive and ordered")
        for role, (mass_min, mass_max) in (
            ("object", self.object_mass_range_kg),
            ("support", self.support_mass_range),
        ):
            if float(mass_min) <= 0 or float(mass_min) > float(mass_max):
                raise ValueError(
                    f"{role}-domain mass range must be positive and ordered"
                )

    @property
    def support_mass_range(self) -> tuple[float, float]:
        """Support mass range, defaulting to the legacy shared mass range."""
        return self.support_mass_range_kg or self.object_mass_range_kg

    def supports(self, module: "ObjectModule") -> bool:
        """Check physical class/features only; labels and colors are ignored."""
        if not isinstance(module, ObjectModule):
            raise TypeError("module must be an ObjectModule")
        return self.supports_object(module.object_spec) and self.supports_support(
            module.support_spec
        )

    def supports_object(self, spec: ObjectSpec) -> bool:
        mass_min, mass_max = self.object_mass_range_kg
        return (
            spec.class_id in self.object_classes
            and spec.geometry in self.object_geometries
            and spec.grasp_mode in self.object_grasp_modes
            and self._inside(spec.size_m, self.object_size_min_m, self.object_size_max_m)
            and float(mass_min) <= float(spec.mass_kg) <= float(mass_max)
            and (self.allow_deformable or not spec.deformable)
        )

    def supports_support(self, spec: ObjectSpec) -> bool:
        mass_min, mass_max = self.support_mass_range
        return (
            spec.class_id in self.support_classes
            and spec.geometry in self.support_geometries
            and spec.support_mode in self.support_modes
            and self._inside(spec.size_m, self.support_size_min_m, self.support_size_max_m)
            and float(mass_min) <= float(spec.mass_kg) <= float(mass_max)
            and (self.allow_deformable or not spec.deformable)
        )

    @staticmethod
    def _inside(
        value: tuple[float, float, float],
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
    ) -> bool:
        return all(float(low) <= float(item) <= float(high) for item, low, high in zip(value, lower, upper))


def rigid_cube_domain(
    *,
    size_min_m: tuple[float, float, float] = (0.04, 0.04, 0.04),
    size_max_m: tuple[float, float, float] = (0.04, 0.04, 0.04),
    mass_range_kg: tuple[float, float] = (0.05, 0.05),
) -> ObjectDomain:
    """Build the role domain for parallel-jaw rigid cube stacking."""
    domain = ObjectDomain(
        object_classes=("cube",),
        support_classes=("cube",),
        object_geometries=("box",),
        support_geometries=("box",),
        object_grasp_modes=("parallel_jaw",),
        support_modes=("flat",),
        object_size_min_m=size_min_m,
        object_size_max_m=size_max_m,
        support_size_min_m=size_min_m,
        support_size_max_m=size_max_m,
        object_mass_range_kg=mass_range_kg,
        support_mass_range_kg=mass_range_kg,
    )
    domain.validate()
    return domain


@dataclass(frozen=True)
class ObjectModule:
    """Role-oriented object pair passed to a manipulation policy."""

    object_label: str
    support_label: str

    def __post_init__(self) -> None:
        if self.object_label == self.support_label:
            raise ValueError("object_label and support_label must differ")
        get_object_spec(self.object_label)
        get_object_spec(self.support_label)

    @property
    def object_spec(self) -> ObjectSpec:
        return get_object_spec(self.object_label)

    @property
    def support_spec(self) -> ObjectSpec:
        return get_object_spec(self.support_label)

    def labels(self) -> tuple[str, str]:
        return self.object_label, self.support_label

    def supports_operation(self, operation: str) -> bool:
        """Check object capability without coupling policy code to labels."""
        if operation == "stack":
            return (
                self.object_spec.stackable
                and self.support_spec.support_mode in {"flat", "concave"}
                and not self.object_spec.deformable
            )
        if operation in {"pick", "place", "transport"}:
            return self.object_spec.grasp_mode != "unknown"
        raise ValueError(f"unsupported operation: {operation!r}")

    def action_context(self) -> dict[str, object]:
        """Return structured object features for future conditioned policies.

        Current frozen policies remain fixed-dimension and use role geometry;
        this context is the stable extension point for object-conditioned
        retraining and is intentionally not converted into raw actions.
        """
        context = self.output().action_context()
        # Compatibility keys for existing object-conditioned datasets.
        context["object_category"] = context["object_class"]
        context["support_category"] = context["support_class"]
        return context

    def action_features(
        self,
        batch_size: int = 1,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Encode normalized physical features without labels or colors.

        New policies may append this vector to their state observation. Old
        fixed-dimension checkpoints remain valid because the runtime only adds
        it when a model explicitly declares this context version.
        """
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        output = self.output()
        values = (
            *(value / 0.1 for value in output.object_size_m),
            *(value / 0.1 for value in output.support_size_m),
            output.object_mass_kg,
            output.support_mass_kg,
            float(output.object_deformable),
            float(output.stackable),
        )
        encoded = torch.tensor(values, device=device, dtype=dtype)
        return encoded.unsqueeze(0).expand(int(batch_size), -1).clone()

    def output(self) -> ObjectModuleOutput:
        """Return one typed result for the action module and diagnostics."""
        obj = self.object_spec
        support = self.support_spec
        return ObjectModuleOutput(
            object_label=self.object_label,
            support_label=self.support_label,
            object_class=obj.class_id,
            support_class=support.class_id,
            object_geometry=obj.geometry,
            support_geometry=support.geometry,
            object_size_m=obj.size_m,
            support_size_m=support.size_m,
            object_mass_kg=obj.mass_kg,
            support_mass_kg=support.mass_kg,
            object_friction_coefficient=obj.friction_coefficient,
            support_friction_coefficient=support.friction_coefficient,
            object_grasp_mode=obj.grasp_mode,
            support_mode=support.support_mode,
            object_deformable=obj.deformable,
            stackable=self.supports_operation("stack"),
            object_color_rgb=obj.color_rgb,
            support_color_rgb=support.color_rgb,
        )

    def role_state(
        self,
        object_position: object,
        support_position: object,
        **state: object,
    ) -> dict[str, object]:
        """Build the color-agnostic state consumed by skill contracts.

        Object identity stays in this module and never becomes a policy
        feature.  The action layer receives only the manipulated object's
        pose as ``object`` and the supporting object's pose as ``support``.
        Extra scalar/proprioceptive fields (for example ``ee`` or ``held``)
        are forwarded unchanged.
        """
        if "object" in state or "support" in state:
            raise ValueError("role_state reserves the object and support fields")
        result: dict[str, object] = dict(state)
        result.update(object=object_position, support=support_position)
        return result

    def resolve_detections(self, detections: Mapping[str, object]) -> tuple[object, object]:
        """Resolve a label-indexed detection map into object/support roles."""
        try:
            return detections[self.object_label], detections[self.support_label]
        except KeyError as exc:
            raise ValueError(f"missing detection for role label: {exc.args[0]!r}") from exc


OBJECT_CATALOG: dict[str, ObjectSpec] = {
    "red_cube": ObjectSpec("red_cube", (220, 40, 40), category="cube", geometry="box", grasp_mode="parallel_jaw", support_mode="flat", stackable=True),
    "blue_cube": ObjectSpec("blue_cube", (40, 90, 220), category="cube", geometry="box", grasp_mode="parallel_jaw", support_mode="flat", stackable=True),
    "green_cube": ObjectSpec("green_cube", (40, 180, 80), category="cube", geometry="box", grasp_mode="parallel_jaw", support_mode="flat", stackable=True),
    "yellow_cube": ObjectSpec("yellow_cube", (220, 190, 40), category="cube", geometry="box", grasp_mode="parallel_jaw", support_mode="flat", stackable=True),
    "white_cup": ObjectSpec("white_cup", (235, 235, 235), (0.08, 0.08, 0.10), category="cup", geometry="cylinder", mass_kg=0.12, grasp_mode="top_rim", support_mode="concave", stackable=False),
    "blue_cup": ObjectSpec("blue_cup", (40, 90, 220), (0.08, 0.08, 0.10), category="cup", geometry="cylinder", mass_kg=0.12, grasp_mode="top_rim", support_mode="concave", stackable=False),
    "folded_shirt": ObjectSpec("folded_shirt", (160, 80, 45), (0.30, 0.25, 0.04), category="cloth", geometry="deformable_mesh", mass_kg=0.20, grasp_mode="surface_clamp", support_mode="deformable", deformable=True, stackable=False),
    "shirt": ObjectSpec("shirt", (160, 80, 45), (0.60, 0.45, 0.01), category="cloth", geometry="deformable_mesh", mass_kg=0.15, grasp_mode="surface_clamp", support_mode="deformable", deformable=True, stackable=False),
    "training_cone": ObjectSpec("training_cone", (180, 80, 210), (0.05, 0.05, 0.04), category="generic", geometry="cone", mass_kg=0.08, grasp_mode="parallel_jaw", support_mode="flat", stackable=False),
    "training_cylinder_object": ObjectSpec("training_cylinder_object", (180, 80, 210), (0.05, 0.05, 0.04), category="generic", geometry="cylinder", mass_kg=0.08, grasp_mode="parallel_jaw", support_mode="flat", stackable=True),
    "training_cylinder_support": ObjectSpec("training_cylinder_support", (40, 90, 220), (0.05, 0.05, 0.04), category="generic", geometry="cylinder", mass_kg=0.08, grasp_mode="parallel_jaw", support_mode="flat", stackable=True),
}


def get_object_spec(label: str) -> ObjectSpec:
    """Return a validated catalog entry for an object label."""
    try:
        return OBJECT_CATALOG[label]
    except KeyError as exc:
        raise ValueError(f"unsupported object label: {label!r}") from exc


def register_object_spec(spec: ObjectSpec, *, overwrite: bool = False) -> ObjectSpec:
    """Register an object type for task/vision modules.

    Registration changes metadata availability only; it does not imply that a
    trained action checkpoint supports the new object.
    """
    if not isinstance(spec, ObjectSpec):
        raise TypeError("spec must be an ObjectSpec")
    if spec.label in OBJECT_CATALOG and not overwrite:
        raise ValueError(f"object label already registered: {spec.label!r}")
    OBJECT_CATALOG[spec.label] = spec
    return spec


def object_labels() -> tuple[str, ...]:
    """Return catalog labels in stable serialization order."""
    return tuple(OBJECT_CATALOG)
