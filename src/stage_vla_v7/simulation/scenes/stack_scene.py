"""Robot, objects, sensors, table, camera, and light for stack tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stage_vla_v7.interfaces import SimulationModelDescriptor

from ..models import (
    CubeModel,
    DepthCameraModel,
    FrankaModel,
    ObserverCameraModel,
    RGBCameraModel,
)
from ..randomization import sample_object_positions


@dataclass(frozen=True)
class SceneLayout:
    seed: int | None
    object_positions_xyz_m: Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True)
class SceneManifest:
    name: str
    robot_id: str
    object_ids: tuple[str, ...]
    sensor_ids: tuple[str, ...]
    includes_table: bool
    includes_light: bool


@dataclass
class StackScene:
    descriptor: SimulationModelDescriptor = SimulationModelDescriptor(
        "red-on-blue-stack-scene",
        "scene",
        "1",
        "backend-neutral",
        ("franka", "rigid-cubes", "rgbd", "seeded-reset"),
    )
    name: str = "red-on-blue-stack"
    robot: FrankaModel = field(default_factory=FrankaModel)
    objects: dict[str, CubeModel] = field(
        default_factory=lambda: {
            "red_cube": CubeModel(color_rgb=(220, 40, 40)),
            "blue_cube": CubeModel(color_rgb=(40, 80, 220)),
            "green_cube": CubeModel(color_rgb=(40, 180, 80)),
        }
    )
    sensors: dict[str, object] = field(
        default_factory=lambda: {
            "rgb": RGBCameraModel(),
            "depth": DepthCameraModel(),
        }
    )
    observer_camera: ObserverCameraModel | None = None
    minimum_separation_m: float = 0.09

    def load(self) -> SceneManifest:
        if "red_cube" not in self.objects or "blue_cube" not in self.objects:
            raise ValueError("stack scene requires red_cube and blue_cube")
        if "rgb" not in self.sensors or "depth" not in self.sensors:
            raise ValueError("stack scene requires RGB and depth sensors")
        sensors = dict(self.sensors)
        if self.observer_camera is not None:
            sensors["observer"] = self.observer_camera
        return SceneManifest(
            self.name,
            self.robot.descriptor.identifier,
            tuple(sorted(model.descriptor.identifier for model in self.objects.values())),
            tuple(sorted(model.descriptor.identifier for model in sensors.values())),
            True,
            True,
        )

    def reset(self, seed: int | None = None) -> SceneLayout:
        self.load()
        return SceneLayout(
            seed,
            sample_object_positions(
                tuple(self.objects),
                seed=seed,
                minimum_separation_m=self.minimum_separation_m,
            ),
        )
