"""Structured V5 manipulation pipeline.

The pipeline is deliberately an orchestration boundary, not another policy.
It connects the semantic VLM/vision output to the existing object, geometry,
load, skill and safety modules while keeping continuous actions at the final
output boundary only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import torch

from .action_output import ActionOutputModule, ObjectActionOutput
from .objects import ObjectModule
from .rl.object_physics import GeometryAdapter, LoadAdapter, PhysicalObjectBatch
from .rl.v5_skill_contracts import ACTION_ORDER, V5Skill
from .stages.grasp_geometry import (
    GraspGeometryProfile,
    grasp_target_position,
    profile_for_geometry,
)
from .task_dsl import (
    SKILL_SEQUENCE,
    InstructionParser,
    InstructionPlan,
    SkillToken,
    StackChainSpec,
    TaskCompiler,
    TaskSpec,
)
from .vision import Detection
from .vlm_interface import VLMOutput


def _position(value: object, *, name: str) -> tuple[float, float, float]:
    """Validate and normalize one metric XYZ position."""
    values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must be a finite XYZ triple")
    return values


@dataclass(frozen=True)
class GeometryPlan:
    """Geometry outputs shared by the scheduler and skill observations."""

    object_position_m: tuple[float, float, float]
    support_position_m: tuple[float, float, float]
    object_yaw_rad: float
    support_yaw_rad: float
    grasp_target_m: tuple[float, float, float]
    lift_target_m: tuple[float, float, float]
    align_target_m: tuple[float, float, float]
    descend_target_m: tuple[float, float, float]
    lift_height_m: float
    stack_center_separation_m: float
    grasp_profile: GraspGeometryProfile

    def validate(self) -> None:
        for name in (
            "object_position_m", "support_position_m", "grasp_target_m",
            "lift_target_m", "align_target_m", "descend_target_m",
        ):
            _position(getattr(self, name), name=name)
        for name in ("object_yaw_rad", "support_yaw_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (self.lift_height_m, self.stack_center_separation_m)
        ):
            raise ValueError("lift height and stack separation must be finite and positive")
        self.grasp_profile.validate()


@dataclass(frozen=True)
class LoadPlan:
    """Physical load limits exposed to the skill scheduler."""

    object_mass_kg: float
    support_mass_kg: float
    friction_coefficient: float
    initial_grip_force_n: float
    maximum_grip_force_n: float
    maximum_speed_mps: float
    maximum_acceleration_mps2: float

    def validate(self) -> None:
        positive = (
            self.object_mass_kg, self.support_mass_kg, self.friction_coefficient,
            self.initial_grip_force_n, self.maximum_grip_force_n,
            self.maximum_speed_mps, self.maximum_acceleration_mps2,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise ValueError("load plan values must be finite and positive")
        if self.initial_grip_force_n > self.maximum_grip_force_n:
            raise ValueError("initial grip force exceeds maximum grip force")


@dataclass(frozen=True)
class ManipulationPlan:
    """Resolved semantics and physical context for one object/support task."""

    task: TaskSpec
    tokens: tuple[SkillToken, ...]
    detections: tuple[Detection, ...]
    object_module: ObjectModule
    geometry: GeometryPlan
    load: LoadPlan
    physical: PhysicalObjectBatch

    def validate(self) -> None:
        if not self.tokens:
            raise ValueError("manipulation plan must contain skill tokens")
        if tuple(token.skill for token in self.tokens) != SKILL_SEQUENCE:
            raise ValueError("manipulation plan must contain the canonical skill sequence")
        self.geometry.validate()
        self.load.validate()

    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(token.skill for token in self.tokens)


class V5ManipulationPipeline:
    """Compose semantic inputs with the frozen physical action boundary.

    ``prepare`` performs no inference and emits no robot action.  ``step`` is
    the only method that can produce a continuous command, and delegates the
    final projection to :class:`ActionOutputModule`.
    """

    def __init__(
        self,
        action_output: ActionOutputModule,
        *,
        compiler: TaskCompiler | None = None,
        instruction_parser: InstructionParser | None = None,
        maximum_speed_mps: float = 0.05,
        maximum_acceleration_mps2: float = 0.5,
        friction_coefficient: float | None = None,
        load_safety_factor: float = 2.0,
        load_min_force_n: float = 5.0,
        load_max_force_n: float = 40.0,
    ) -> None:
        if not isinstance(action_output, ActionOutputModule):
            raise TypeError("action_output must be an ActionOutputModule")
        for name, value in (
            ("maximum_speed_mps", maximum_speed_mps),
            ("maximum_acceleration_mps2", maximum_acceleration_mps2),
            ("load_safety_factor", load_safety_factor),
            ("load_min_force_n", load_min_force_n),
            ("load_max_force_n", load_max_force_n),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if load_min_force_n > load_max_force_n:
            raise ValueError("load_min_force_n must not exceed load_max_force_n")
        if friction_coefficient is not None and (
            not math.isfinite(float(friction_coefficient)) or float(friction_coefficient) <= 0.0
        ):
            raise ValueError("friction_coefficient must be finite and positive")
        self.action_output = action_output
        self.compiler = compiler or TaskCompiler()
        self.instruction_parser = instruction_parser or InstructionParser(
            compiler=self.compiler
        )
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self.friction_coefficient = friction_coefficient
        self.load_safety_factor = float(load_safety_factor)
        self.load_min_force_n = float(load_min_force_n)
        self.load_max_force_n = float(load_max_force_n)

    def prepare_from_vlm(self, output: VLMOutput) -> ManipulationPlan:
        """Resolve a VLM result without allowing it to bypass the DSL."""
        if not isinstance(output, VLMOutput):
            raise TypeError("output must be a VLMOutput")
        return self.prepare(output.task, output.detections)

    def prepare_from_vlm_request(self, adapter: object, request: object) -> ManipulationPlan:
        """Run a replaceable VLM adapter, then resolve only its semantics."""
        predict = getattr(adapter, "predict", None)
        if not callable(predict):
            raise TypeError("adapter must provide predict(request)")
        return self.prepare_from_vlm(predict(request))

    def compile_instruction(self, command: str) -> InstructionPlan:
        """Compile instruction text without invoking a language model.

        This method is semantic only: it returns typed tasks and skill tokens,
        never a continuous robot action.  Object properties are resolved later
        by :meth:`prepare` through ``ObjectModule``.
        """
        return self.instruction_parser.plan(command)

    def prepare_from_instruction(
        self,
        command: str,
        detections: Iterable[Detection],
    ) -> ManipulationPlan:
        """Prepare a single instruction after deterministic parsing.

        Multi-relation commands are available through ``compile_instruction``;
        execution of each relation still belongs to the chain scheduler so
        physical state is not accidentally reset or replayed here.
        """
        parsed = self.compile_instruction(command)
        if isinstance(parsed.task, StackChainSpec):
            raise ValueError(
                "prepare_from_instruction accepts one relation; use "
                "compile_instruction for a multi-relation action chain"
            )
        return self.prepare(parsed.task, detections)

    def prepare(
        self,
        task: TaskSpec,
        detections: Iterable[Detection],
    ) -> ManipulationPlan:
        """Build object, geometry and load context from structured semantics."""
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        checked: list[Detection] = []
        by_label: dict[str, Detection] = {}
        for detection in detections:
            if not isinstance(detection, Detection):
                raise TypeError("detections must contain Detection values")
            detection.validate()
            if detection.label in by_label:
                raise ValueError(f"duplicate detection label: {detection.label}")
            by_label[detection.label] = detection
            checked.append(detection)

        module = ObjectModule(task.object_name, task.target_name)
        try:
            object_detection = by_label[module.object_label]
            support_detection = by_label[module.support_label]
        except KeyError as exc:
            raise ValueError(f"missing detection for role label: {exc.args[0]!r}") from exc
        if not module.supports_operation("stack"):
            raise ValueError("object/support pair is not stackable")
        if module.object_spec.grasp_mode != "parallel_jaw":
            raise ValueError(
                "the current geometry/action bundle requires parallel_jaw grasp mode"
            )

        object_position = _position(object_detection.position_xyz_m, name="object_position_m")
        support_position = _position(support_detection.position_xyz_m, name="support_position_m")
        object_tensor = torch.tensor([object_position], dtype=torch.float32)
        object_size = torch.tensor([module.object_spec.size_m], dtype=torch.float32)
        profile = profile_for_geometry(module.object_spec.geometry)
        grasp_target = grasp_target_position(object_tensor, object_size, profile=profile)[0]
        physical = PhysicalObjectBatch(
            object_size_m=object_size,
            support_size_m=torch.tensor([module.support_spec.size_m], dtype=torch.float32),
            object_mass_kg=torch.tensor([module.object_spec.mass_kg]),
            support_mass_kg=torch.tensor([module.support_spec.mass_kg]),
            object_deformable=torch.tensor([module.object_spec.deformable]),
            stackable=torch.tensor([module.supports_operation("stack")]),
            geometry_bundle=f"{module.object_spec.geometry}_{module.object_spec.grasp_mode}",
        )
        geometry_adapter = GeometryAdapter(physical)
        separation = float(geometry_adapter.stack_center_separation_m[0])
        lift_height = float(max(0.06, separation + 0.02))
        lift_target = (object_position[0], object_position[1], object_position[2] + lift_height)
        align_target = (
            support_position[0], support_position[1], support_position[2] + separation
        )
        geometry = GeometryPlan(
            object_position_m=object_position,
            support_position_m=support_position,
            object_yaw_rad=float(object_detection.yaw_rad),
            support_yaw_rad=float(support_detection.yaw_rad),
            grasp_target_m=tuple(float(item) for item in grasp_target),
            lift_target_m=lift_target,
            align_target_m=align_target,
            descend_target_m=align_target,
            lift_height_m=lift_height,
            stack_center_separation_m=separation,
            grasp_profile=profile,
        )
        friction = (
            float(self.friction_coefficient)
            if self.friction_coefficient is not None
            else float(module.object_spec.friction_coefficient)
        )
        load_adapter = LoadAdapter(
            physical,
            friction_coefficient=friction,
            safety_factor=self.load_safety_factor,
            min_force_n=self.load_min_force_n,
            max_force_n=self.load_max_force_n,
        )
        load = LoadPlan(
            object_mass_kg=float(module.object_spec.mass_kg),
            support_mass_kg=float(module.support_spec.mass_kg),
            friction_coefficient=friction,
            initial_grip_force_n=float(load_adapter.initial_force_target_n[0]),
            maximum_grip_force_n=float(self.load_max_force_n),
            maximum_speed_mps=self.maximum_speed_mps,
            maximum_acceleration_mps2=self.maximum_acceleration_mps2,
        )
        plan = ManipulationPlan(
            task=task,
            tokens=self.compiler.compile_task(task),
            detections=tuple(checked),
            object_module=module,
            geometry=geometry,
            load=load,
            physical=physical,
        )
        plan.validate()
        return plan

    def step(
        self,
        plan: ManipulationPlan,
        skill: V5Skill | str,
        observation: torch.Tensor,
        *,
        finished: torch.Tensor | None = None,
        reference_state: Mapping[str, object] | None = None,
        include_reference: bool = False,
    ) -> ObjectActionOutput:
        """Run one selected skill and return the safety-projected command."""
        if not isinstance(plan, ManipulationPlan):
            raise TypeError("plan must be a ManipulationPlan")
        plan.validate()
        selected = V5Skill(skill)
        if selected.value not in plan.skills:
            raise ValueError(f"skill {selected.value} is not present in the task plan")
        return self.action_output.emit_for(
            plan.object_module,
            selected,
            observation,
            finished=finished,
            reference_state=reference_state,
            include_reference=include_reference,
        )


__all__ = [
    "GeometryPlan",
    "LoadPlan",
    "ManipulationPlan",
    "V5ManipulationPipeline",
]
