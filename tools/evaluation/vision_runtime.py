"""Vectorized Vision observation with per-environment strict isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from stage_vla_v7.vision import VisionRequest

from .vision_policy import VisionFailPolicy, VisionPositionResolver


@dataclass
class VisionBatchObserver:
    """Observe only live environments and keep dead rows shape-safe."""

    camera: Any
    camera_adapter: Any
    service: Any
    resolver: VisionPositionResolver
    origins: Any
    scene_assets: Sequence[str]
    labels: Mapping[str, str]
    camera_name: str
    device: Any
    stats: MutableMapping[str, object]
    alive: np.ndarray
    debug_oracle_loader: Callable[[str], np.ndarray] | None = None

    def observe_positions(self) -> tuple[dict[str, Any], np.ndarray]:
        import torch

        active = np.asarray(self.alive, dtype=bool).copy()
        predicted = {
            name: np.full((self.resolver.num_envs, 3), np.nan, dtype=np.float32)
            for name in self.scene_assets
        }
        if bool(active.any()):
            rgb = self.camera_adapter.rgb_u8_batch(
                self.camera, camera_name=self.camera_name
            )
            depth = self.camera_adapter.depth_m_batch(
                self.camera, camera_name=self.camera_name
            )
            calls = self.stats["v7_service_calls_by_environment"]
            for env_index in np.flatnonzero(active).tolist():
                observed = self.service.observe(VisionRequest(
                    rgb=rgb[env_index],
                    depth_m=depth[env_index],
                    frame_id=f"isaac-env-{env_index}",
                    metadata={"environment_index": env_index},
                ))
                calls[env_index] += 1
                self.stats["v7_service_calls"] = int(
                    self.stats["v7_service_calls"]
                ) + 1
                by_label = {item.label: item for item in observed.scene.detections}
                for asset_name in self.scene_assets:
                    detection = by_label.get(self.labels[asset_name])
                    if detection is not None:
                        predicted[asset_name][env_index] = np.asarray(
                            detection.position_xyz_m, dtype=np.float32
                        )
            self.stats["frames"] = int(self.stats["frames"]) + int(active.sum())

        if self.resolver.policy is VisionFailPolicy.STRICT:
            batch = self.resolver.resolve_isolated(
                predicted,
                active_mask=active,
            )
            resolved = batch.positions
            valid = batch.valid_mask
            self.alive[:] = batch.alive_mask
        else:
            resolved, valid = self.resolver.resolve(
                predicted,
                oracle_loader=self.debug_oracle_loader,
            )
            self.alive[:] &= valid

        self.stats.update(self.resolver.audit())
        calls = self.stats["v7_service_calls_by_environment"]
        for row in self.stats["per_environment"]:
            row["service_calls"] = int(calls[row["environment_index"]])
        positions = {
            asset_name: (
                torch.as_tensor(local_xyz, device=self.device, dtype=torch.float32)
                + self.origins
            )
            for asset_name, local_xyz in resolved.items()
        }
        return positions, self.alive.copy()


__all__ = ["VisionBatchObserver"]
