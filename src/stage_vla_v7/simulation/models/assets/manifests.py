"""External simulator asset records; large USD files stay outside Git."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationAssetManifest:
    logical_id: str
    expected_reference: str
    version: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.logical_id.strip() or not self.expected_reference.strip() or not self.version.strip():
            raise ValueError("simulation asset manifest fields must be non-empty")
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ValueError("asset SHA256 must contain 64 hexadecimal characters")


FRANKA_ASSET = SimulationAssetManifest(
    "isaaclab.franka.panda",
    "Isaac-Stack-Cube-Franka-IK-Rel-v0:scene.robot",
    "Isaac Lab registered task",
)
