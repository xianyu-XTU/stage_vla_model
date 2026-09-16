"""Simulator-neutral adapter implementation for Isaac Lab payloads."""

from __future__ import annotations

from stage_vla_v7.interfaces import (
    RobotAction,
    RobotObservation,
    SimulationAction,
    SimulationModelDescriptor,
    SimulationObservation,
    VisionRequest,
)

from .action_adapter import IsaacActionAdapter
from .camera_adapter import IsaacCameraAdapter


class IsaacLabAdapter:
    descriptor = SimulationModelDescriptor(
        "isaac-lab",
        "simulation-adapter",
        "1",
        "isaac-lab",
        ("rgbd", "relative-cartesian-5d", "batched-torch"),
    )

    def __init__(
        self,
        *,
        camera: IsaacCameraAdapter | None = None,
        action: IsaacActionAdapter | None = None,
    ) -> None:
        self.camera = camera or IsaacCameraAdapter()
        self.action = action or IsaacActionAdapter()

    def to_vision_request(self, observation: SimulationObservation) -> VisionRequest:
        return self.camera.to_vision_request(
            observation.rgb,
            observation.depth_m,
            frame_id=observation.frame_id,
            timestamp_s=observation.timestamp_s,
            metadata=observation.metadata,
        )

    def to_robot_observation(self, observation: SimulationObservation) -> RobotObservation:
        return observation.robot

    def to_simulation_action(self, action: RobotAction) -> SimulationAction:
        return self.action.to_simulation_action(action)
