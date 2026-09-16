"""Vision port definitions. This module has no action or language dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, runtime_checkable

from stage_vla_v7.contracts import ModelDescriptor, SceneState


@dataclass(frozen=True)
class VisionRequest:
    """Opaque camera payload accepted by a vision provider."""

    rgb: object
    depth_m: object | None = None
    frame_id: str | None = None
    timestamp_s: float | None = None
    calibration: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rgb is None:
            raise ValueError("rgb payload is required")
        if self.timestamp_s is not None and not math.isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")


@dataclass(frozen=True)
class VisionResult:
    """Validated scene plus provider audit information."""

    scene: SceneState
    provider: ModelDescriptor
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class VisionProvider(Protocol):
    """Replaceable detector boundary."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    def detect(self, request: VisionRequest) -> VisionResult:
        """Convert one camera request into a metric scene state."""
