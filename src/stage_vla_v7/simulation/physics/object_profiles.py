"""Label-free per-environment physical profiles and controller adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


# Frozen snapshot/action-context ABI. The historical value must not change.
ACTION_CONTEXT_VERSION = "stage_vla_v5.object_physics.v1"
ACTION_CONTEXT_ORDER = (
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "support_size_x",
    "support_size_y",
    "support_size_z",
    "object_mass",
    "support_mass",
    "object_deformable",
    "stackable",
)


def snapshot_has_physical_context(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("physical_context_version") == ACTION_CONTEXT_VERSION
        and "physical_context" in payload
    )


def _as_float_tensor(value: Tensor | object, *, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _range3(value: object, *, name: str) -> tuple[tuple[float, float], ...]:
    rows = tuple(tuple(float(item) for item in row) for row in value)  # type: ignore[arg-type]
    if len(rows) != 3 or any(len(row) != 2 for row in rows):
        raise ValueError(f"{name} must contain three [min,max] ranges")
    if any(low <= 0 or low > high for low, high in rows):
        raise ValueError(f"{name} ranges must be positive and ordered")
    return rows


def _range1(value: object, *, name: str) -> tuple[float, float]:
    result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(result) != 2 or result[0] <= 0 or result[0] > result[1]:
        raise ValueError(f"{name} must be a positive ordered [min,max] range")
    return result


@dataclass(frozen=True)
class PhysicalDomainConfig:
    geometry_bundle: str
    object_size_range_m: tuple[tuple[float, float], ...]
    support_size_range_m: tuple[tuple[float, float], ...]
    object_mass_range_kg: tuple[float, float]
    support_mass_range_kg: tuple[float, float]
    object_deformable: bool = False
    stackable: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PhysicalDomainConfig":
        geometry_bundle = str(value.get("geometry_bundle", "")).strip()
        if not geometry_bundle:
            raise ValueError("geometry_bundle must be a non-empty string")
        return cls(
            geometry_bundle=geometry_bundle,
            object_size_range_m=_range3(
                value["object_size_range_m"], name="object_size_range_m"
            ),
            support_size_range_m=_range3(
                value["support_size_range_m"], name="support_size_range_m"
            ),
            object_mass_range_kg=_range1(
                value["object_mass_range_kg"], name="object_mass_range_kg"
            ),
            support_mass_range_kg=_range1(
                value["support_mass_range_kg"], name="support_mass_range_kg"
            ),
            object_deformable=bool(value.get("object_deformable", False)),
            stackable=bool(value.get("stackable", True)),
        )

    def sample(self, num_envs: int, *, seed: int) -> "PhysicalObjectBatch":
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        def sample_ranges(ranges: tuple[tuple[float, float], ...]) -> Tensor:
            bounds = torch.tensor(ranges, dtype=torch.float32)
            unit = torch.rand((int(num_envs), len(ranges)), generator=generator)
            return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])

        def sample_range(bounds: tuple[float, float]) -> Tensor:
            unit = torch.rand((int(num_envs),), generator=generator)
            return float(bounds[0]) + unit * float(bounds[1] - bounds[0])

        return PhysicalObjectBatch(
            object_size_m=sample_ranges(self.object_size_range_m),
            support_size_m=sample_ranges(self.support_size_range_m),
            object_mass_kg=sample_range(self.object_mass_range_kg),
            support_mass_kg=sample_range(self.support_mass_range_kg),
            object_deformable=torch.full(
                (int(num_envs),), self.object_deformable, dtype=torch.bool
            ),
            stackable=torch.full(
                (int(num_envs),), self.stackable, dtype=torch.bool
            ),
            geometry_bundle=self.geometry_bundle,
        )


@dataclass(frozen=True)
class PhysicalObjectBatch:
    object_size_m: Tensor
    support_size_m: Tensor
    object_mass_kg: Tensor
    support_mass_kg: Tensor
    object_deformable: Tensor
    stackable: Tensor
    geometry_bundle: str = "box_parallel_jaw"
    context_version: str = ACTION_CONTEXT_VERSION

    def __post_init__(self) -> None:
        object_size = _as_float_tensor(self.object_size_m, name="object_size_m")
        support_size = _as_float_tensor(self.support_size_m, name="support_size_m")
        object_mass = _as_float_tensor(self.object_mass_kg, name="object_mass_kg")
        support_mass = _as_float_tensor(self.support_mass_kg, name="support_mass_kg")
        deformable = torch.as_tensor(self.object_deformable, dtype=torch.bool)
        stackable = torch.as_tensor(self.stackable, dtype=torch.bool)
        if object_size.ndim != 2 or object_size.shape[1] != 3:
            raise ValueError("object_size_m must have shape [N,3]")
        count = object_size.shape[0]
        if support_size.shape != (count, 3):
            raise ValueError("support_size_m must have shape [N,3]")
        if object_mass.shape != (count,) or support_mass.shape != (count,):
            raise ValueError("mass tensors must have shape [N]")
        if deformable.shape != (count,) or stackable.shape != (count,):
            raise ValueError("capability tensors must have shape [N]")
        if count < 1 or torch.any(object_size <= 0) or torch.any(support_size <= 0):
            raise ValueError("physical batch sizes must be positive")
        if torch.any(object_mass <= 0) or torch.any(support_mass <= 0):
            raise ValueError("physical batch masses must be positive")
        if not self.geometry_bundle.strip():
            raise ValueError("geometry_bundle must be a non-empty string")
        if self.context_version != ACTION_CONTEXT_VERSION:
            raise ValueError(
                f"unsupported physical context version: {self.context_version!r}"
            )
        object.__setattr__(self, "object_size_m", object_size)
        object.__setattr__(self, "support_size_m", support_size)
        object.__setattr__(self, "object_mass_kg", object_mass)
        object.__setattr__(self, "support_mass_kg", support_mass)
        object.__setattr__(self, "object_deformable", deformable)
        object.__setattr__(self, "stackable", stackable)

    @classmethod
    def from_scalar_config(
        cls,
        config,
        num_envs: int,
        *,
        support_size_m: tuple[float, float, float] | None = None,
        support_mass_kg: float | None = None,
        geometry_bundle: str = "box_parallel_jaw",
    ) -> "PhysicalObjectBatch":
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        object_size = torch.tensor(
            [config.width_m, config.depth_m, config.height_m], dtype=torch.float32
        ).expand(int(num_envs), -1).clone()
        support_size = torch.tensor(
            support_size_m or (config.width_m, config.depth_m, config.height_m),
            dtype=torch.float32,
        ).expand(int(num_envs), -1).clone()
        return cls(
            object_size_m=object_size,
            support_size_m=support_size,
            object_mass_kg=torch.full((int(num_envs),), float(config.mass_kg)),
            support_mass_kg=torch.full(
                (int(num_envs),),
                float(config.mass_kg if support_mass_kg is None else support_mass_kg),
            ),
            object_deformable=torch.zeros(int(num_envs), dtype=torch.bool),
            stackable=torch.ones(int(num_envs), dtype=torch.bool),
            geometry_bundle=geometry_bundle,
        )

    @classmethod
    def from_action_context(
        cls,
        context: Tensor | object,
        *,
        geometry_bundle: str,
        context_version: str = ACTION_CONTEXT_VERSION,
    ) -> "PhysicalObjectBatch":
        value = _as_float_tensor(context, name="physical_context")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != 10:
            raise ValueError("physical_context must have shape [N,10]")
        flags = value[:, 8:10]
        if not torch.all((flags == 0.0) | (flags == 1.0)):
            raise ValueError("physical context capability flags must be binary")
        return cls(
            object_size_m=value[:, :3] * 0.1,
            support_size_m=value[:, 3:6] * 0.1,
            object_mass_kg=value[:, 6],
            support_mass_kg=value[:, 7],
            object_deformable=flags[:, 0].bool(),
            stackable=flags[:, 1].bool(),
            geometry_bundle=geometry_bundle,
            context_version=context_version,
        )

    @classmethod
    def from_snapshot_payloads(
        cls,
        payloads: list[Mapping[str, object]],
        num_envs: int,
    ) -> "PhysicalObjectBatch":
        if not payloads or int(num_envs) < 1:
            raise ValueError("snapshot payloads and num_envs must be non-empty")
        if not all(snapshot_has_physical_context(payload) for payload in payloads):
            raise ValueError("all snapshots must contain the current physical context")
        bundles = {
            str(payload.get("geometry_bundle", "")).strip() for payload in payloads
        }
        if len(bundles) != 1 or not next(iter(bundles)):
            raise ValueError(
                "snapshot geometry_bundle values must match and be non-empty"
            )
        rows = [
            payloads[index % len(payloads)]["physical_context"]
            for index in range(int(num_envs))
        ]
        return cls.from_action_context(
            rows,
            geometry_bundle=next(iter(bundles)),
            context_version=str(payloads[0]["physical_context_version"]),
        )

    @property
    def num_envs(self) -> int:
        return int(self.object_size_m.shape[0])

    def to(self, device: torch.device | str) -> "PhysicalObjectBatch":
        return PhysicalObjectBatch(
            object_size_m=self.object_size_m.to(device),
            support_size_m=self.support_size_m.to(device),
            object_mass_kg=self.object_mass_kg.to(device),
            support_mass_kg=self.support_mass_kg.to(device),
            object_deformable=self.object_deformable.to(device),
            stackable=self.stackable.to(device),
            geometry_bundle=self.geometry_bundle,
            context_version=self.context_version,
        )

    def action_context(self) -> Tensor:
        return torch.cat(
            [
                self.object_size_m / 0.1,
                self.support_size_m / 0.1,
                self.object_mass_kg.unsqueeze(-1),
                self.support_mass_kg.unsqueeze(-1),
                self.object_deformable.float().unsqueeze(-1),
                self.stackable.float().unsqueeze(-1),
            ],
            dim=-1,
        )


@dataclass(frozen=True)
class GeometryAdapter:
    physical: PhysicalObjectBatch
    grasp_width_ratio: float = 1.0
    jaw_clearance_m: float = 0.002
    max_compression_m: float = 0.004
    joint_min_m: float = 0.0
    joint_max_m: float = 0.04

    def __post_init__(self) -> None:
        if not 0 < float(self.grasp_width_ratio) <= 1:
            raise ValueError("grasp_width_ratio must lie in (0,1]")
        if min(
            float(self.jaw_clearance_m),
            float(self.max_compression_m),
            float(self.joint_min_m),
        ) < 0:
            raise ValueError(
                "jaw clearance, compression and joint minimum must be non-negative"
            )
        if float(self.joint_min_m) >= float(self.joint_max_m):
            raise ValueError("joint_min_m must be less than joint_max_m")
        if torch.any(
            self.effective_width_m / 2 + float(self.jaw_clearance_m)
            > float(self.joint_max_m)
        ):
            raise ValueError("object width plus clearance exceeds gripper opening")

    @property
    def effective_width_m(self) -> Tensor:
        planar = self.physical.object_size_m[:, :2]
        return planar.norm(dim=-1) * float(self.grasp_width_ratio)

    @property
    def minimum_planar_width_m(self) -> Tensor:
        return self.physical.object_size_m[:, :2].amin(dim=-1) * float(
            self.grasp_width_ratio
        )

    @property
    def geometric_joint_target_m(self) -> Tensor:
        return (
            self.effective_width_m / 2 + float(self.jaw_clearance_m)
        ).clamp_max(float(self.joint_max_m))

    @property
    def compression_joint_min_m(self) -> Tensor:
        return (
            self.minimum_planar_width_m / 2 - float(self.max_compression_m)
        ).clamp_min(float(self.joint_min_m))

    @property
    def stack_center_separation_m(self) -> Tensor:
        return (
            self.physical.object_size_m[:, 2]
            + self.physical.support_size_m[:, 2]
        ) / 2

    def legacy_size_features(self) -> Tensor:
        return self.physical.object_size_m / 0.1

    def pressure_feedback_step(
        self,
        current_joint_pos: Tensor,
        measured_force_n: Tensor,
        target_force_n: Tensor,
        *,
        gain_m_per_n: float,
        max_step_m: float,
    ) -> Tensor:
        current = torch.as_tensor(current_joint_pos)
        measured = torch.as_tensor(
            measured_force_n, device=current.device, dtype=current.dtype
        )
        target = torch.as_tensor(
            target_force_n, device=current.device, dtype=current.dtype
        )
        if (
            current.shape != (self.physical.num_envs, 2)
            or measured.shape != current.shape
        ):
            raise ValueError("joint positions and forces must have shape [N,2]")
        if target.shape != (self.physical.num_envs,):
            raise ValueError("target_force_n must have shape [N]")
        if not torch.isfinite(current).all() or not torch.isfinite(measured).all():
            raise ValueError("joint positions and forces must be finite")
        if torch.any(measured < 0) or gain_m_per_n <= 0 or max_step_m <= 0:
            raise ValueError("forces, feedback gain and max step must be valid")
        lower = self.compression_joint_min_m.to(current).unsqueeze(-1)
        upper = torch.full_like(lower, float(self.joint_max_m))
        base = torch.maximum(lower, torch.minimum(current, upper))
        error = target.unsqueeze(-1) - measured
        delta = (-float(gain_m_per_n) * error).clamp(
            -float(max_step_m), float(max_step_m)
        )
        return torch.maximum(lower, torch.minimum(base + delta, upper))


@dataclass(frozen=True)
class LoadAdapter:
    physical: PhysicalObjectBatch
    friction_coefficient: float = 0.4
    safety_factor: float = 2.0
    lift_acceleration_mps2: float = 0.5
    min_force_n: float = 5.0
    max_force_n: float = 40.0
    residual_force_range_n: float = 4.0
    residual_deadband: float = 0.0
    pressure_tolerance_n: float = 1.0
    force_balance_tolerance_n: float = 1.5

    def __post_init__(self) -> None:
        positive = (
            self.friction_coefficient,
            self.safety_factor,
            self.min_force_n,
            self.max_force_n,
            self.pressure_tolerance_n,
            self.force_balance_tolerance_n,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("friction, force and tolerance parameters must be positive")
        if self.min_force_n > self.max_force_n:
            raise ValueError("min_force_n must not exceed max_force_n")
        if self.lift_acceleration_mps2 < 0 or self.residual_force_range_n < 0:
            raise ValueError("acceleration and residual force range must be non-negative")
        if not 0 <= float(self.residual_deadband) < 1:
            raise ValueError("residual_deadband must lie in [0,1)")

    @property
    def initial_force_target_n(self) -> Tensor:
        required = (
            float(self.safety_factor)
            * self.physical.object_mass_kg
            * (9.81 + float(self.lift_acceleration_mps2))
            / (2 * float(self.friction_coefficient))
        )
        return required.clamp(float(self.min_force_n), float(self.max_force_n))

    def target_from_grip(self, grip: Tensor) -> Tensor:
        value = torch.as_tensor(grip, device=self.physical.object_mass_kg.device)
        if value.shape != (self.physical.num_envs,) or not torch.isfinite(value).all():
            raise ValueError("grip must be finite with shape [N]")
        closure = (-value).clamp(0.0, 1.0)
        closure = (
            (closure - float(self.residual_deadband))
            / (1.0 - float(self.residual_deadband))
        ).clamp(0.0, 1.0)
        return (
            self.initial_force_target_n
            + closure * float(self.residual_force_range_n)
        ).clamp(float(self.min_force_n), float(self.max_force_n))

    def tracking_ok(
        self, measured_force_n: Tensor, target_force_n: Tensor
    ) -> Tensor:
        measured = torch.as_tensor(measured_force_n)
        target = torch.as_tensor(
            target_force_n, device=measured.device, dtype=measured.dtype
        )
        if measured.shape != (self.physical.num_envs, 2):
            raise ValueError("measured_force_n must have shape [N,2]")
        if target.shape != (self.physical.num_envs,):
            raise ValueError("target_force_n must have shape [N]")
        if not torch.isfinite(measured).all() or torch.any(measured < 0):
            raise ValueError("measured force must be finite and non-negative")
        error_ok = (
            (measured - target.unsqueeze(-1)).abs().amax(dim=-1)
            <= float(self.pressure_tolerance_n)
        )
        balance_ok = (
            (measured[:, 0] - measured[:, 1]).abs()
            <= float(self.force_balance_tolerance_n)
        )
        return error_ok & balance_ok
