"""Fail-closed handling for RGB-D object-position observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np


class VisionFailPolicy(str, Enum):
    STRICT = "strict"
    DEBUG_ORACLE = "debug_oracle"


class VisionDetectionError(RuntimeError):
    """Raised when required RGB-D detections cannot produce finite positions."""

    def __init__(
        self,
        *,
        observation_index: int,
        failed_objects: Sequence[str],
        failed_environments: Sequence[int],
        policy: VisionFailPolicy,
    ) -> None:
        self.observation_index = int(observation_index)
        self.failed_objects = tuple(sorted(set(failed_objects)))
        self.failed_environments = tuple(sorted(set(int(i) for i in failed_environments)))
        self.policy = policy
        super().__init__(
            "RGB-D detection failed under "
            f"{policy.value!r} policy at observation {observation_index}: "
            f"objects={list(self.failed_objects)!r}, "
            f"environments={list(self.failed_environments)!r}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": "VISION",
            "reason": "invalid_required_detection",
            "fail_policy": self.policy.value,
            "observation_index": self.observation_index,
            "failed_objects": list(self.failed_objects),
            "failed_environments": list(self.failed_environments),
        }


@dataclass
class VisionPositionResolver:
    """Resolve detector positions without hiding missing RGB-D observations."""

    policy: VisionFailPolicy
    asset_names: tuple[str, ...]
    num_envs: int
    tracked: dict[str, np.ndarray] = field(default_factory=dict)
    observation_count: int = 0
    invalid_frames: int = 0
    missing_by_asset: dict[str, int] = field(init=False)
    failed_objects: set[str] = field(default_factory=set)
    oracle_fallback_count: int = 0
    oracle_fallback_objects: set[str] = field(default_factory=set)
    oracle_fallback_steps: set[int] = field(default_factory=set)
    oracle_fallback_events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.asset_names = tuple(self.asset_names)
        self.num_envs = int(self.num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not self.asset_names or len(set(self.asset_names)) != len(self.asset_names):
            raise ValueError("asset_names must be non-empty and unique")
        self.missing_by_asset = {name: 0 for name in self.asset_names}

    def resolve(
        self,
        predicted: Mapping[str, np.ndarray],
        *,
        oracle_loader: Callable[[str], np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return finite positions or fail before strict mode touches oracle state."""
        observation_index = self.observation_count
        self.observation_count += 1
        arrays: dict[str, np.ndarray] = {}
        observed_by_asset: dict[str, np.ndarray] = {}
        current_valid = np.ones(self.num_envs, dtype=bool)

        for asset_name in self.asset_names:
            if asset_name not in predicted:
                raise ValueError(f"missing predicted asset {asset_name!r}")
            values = np.asarray(predicted[asset_name], dtype=np.float32)
            if values.shape != (self.num_envs, 3):
                raise ValueError(
                    f"predicted {asset_name!r} must have shape ({self.num_envs}, 3)"
                )
            arrays[asset_name] = values.copy()
            observed = np.isfinite(values).all(axis=1)
            observed_by_asset[asset_name] = observed
            missing_count = int((~observed).sum())
            self.missing_by_asset[asset_name] += missing_count
            if missing_count:
                self.failed_objects.add(asset_name)
            current_valid &= observed

        self.invalid_frames += int((~current_valid).sum())
        if self.policy is VisionFailPolicy.STRICT and not bool(current_valid.all()):
            failed_assets = [
                name for name, observed in observed_by_asset.items()
                if not bool(observed.all())
            ]
            failed_envs = np.flatnonzero(~current_valid).tolist()
            raise VisionDetectionError(
                observation_index=observation_index,
                failed_objects=failed_assets,
                failed_environments=failed_envs,
                policy=self.policy,
            )

        resolved: dict[str, np.ndarray] = {}
        resolved_valid = np.ones(self.num_envs, dtype=bool)
        for asset_name, values in arrays.items():
            observed = observed_by_asset[asset_name]
            previous = self.tracked.get(asset_name)
            if previous is not None:
                previous_valid = np.isfinite(previous).all(axis=1)
                use_previous = ~observed & previous_valid
                values[use_previous] = previous[use_previous]

            unresolved = ~np.isfinite(values).all(axis=1)
            if bool(unresolved.any()):
                if self.policy is not VisionFailPolicy.DEBUG_ORACLE:
                    raise AssertionError("strict vision reached debug fallback")
                if oracle_loader is None:
                    raise VisionDetectionError(
                        observation_index=observation_index,
                        failed_objects=(asset_name,),
                        failed_environments=np.flatnonzero(unresolved).tolist(),
                        policy=self.policy,
                    )
                oracle_values = np.asarray(oracle_loader(asset_name), dtype=np.float32)
                if oracle_values.shape != (self.num_envs, 3):
                    raise ValueError(
                        f"oracle {asset_name!r} must have shape ({self.num_envs}, 3)"
                    )
                oracle_valid = np.isfinite(oracle_values).all(axis=1)
                fallback_mask = unresolved & oracle_valid
                values[fallback_mask] = oracle_values[fallback_mask]
                for environment_index in np.flatnonzero(fallback_mask).tolist():
                    self.oracle_fallback_events.append({
                        "observation_index": observation_index,
                        "environment_index": int(environment_index),
                        "object": asset_name,
                    })
                fallback_count = int(fallback_mask.sum())
                self.oracle_fallback_count += fallback_count
                if fallback_count:
                    self.oracle_fallback_objects.add(asset_name)
                    self.oracle_fallback_steps.add(observation_index)

            finite = np.isfinite(values).all(axis=1)
            resolved_valid &= finite
            if not bool(finite.all()):
                raise VisionDetectionError(
                    observation_index=observation_index,
                    failed_objects=(asset_name,),
                    failed_environments=np.flatnonzero(~finite).tolist(),
                    policy=self.policy,
                )
            if previous is None:
                self.tracked[asset_name] = arrays[asset_name].copy()
            else:
                self.tracked[asset_name] = np.where(
                    observed[:, None], arrays[asset_name], previous
                )
            resolved[asset_name] = values
        return resolved, resolved_valid

    def audit(self) -> dict[str, object]:
        return {
            "fail_policy": self.policy.value,
            "strict_mode": self.policy is VisionFailPolicy.STRICT,
            "invalid_frames": self.invalid_frames,
            "missing_by_asset": dict(self.missing_by_asset),
            "failed_objects": sorted(self.failed_objects),
            "oracle_fallback_used": self.oracle_fallback_count > 0,
            "oracle_fallback_count": self.oracle_fallback_count,
            "oracle_fallback_objects": sorted(self.oracle_fallback_objects),
            "oracle_fallback_steps": sorted(self.oracle_fallback_steps),
            "oracle_fallback_events": list(self.oracle_fallback_events),
        }


__all__ = [
    "VisionDetectionError",
    "VisionFailPolicy",
    "VisionPositionResolver",
]
