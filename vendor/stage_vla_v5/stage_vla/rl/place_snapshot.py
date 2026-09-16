"""Portable PLACE snapshot helpers for M17.

A snapshot is captured from a *single* validated Isaac Lab environment with
``scene.get_state(is_relative=True)`` and stored on CPU.  At training time the
same relative scene state is expanded to any number of cloned environments and
restored with ``env.reset_to(..., is_relative=True)``.  This avoids assuming
that a raw-action trace recorded in a 1-env scene can be replayed unchanged in
a vectorized scene whose clones have different ``env_origins``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

SNAPSHOT_VERSION = "M17-place-snapshot-v2"


def _map_tensors(value: Any, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    return deepcopy(value)


def single_env_cpu_state(scene_state: dict, env_id: int = 0) -> dict:
    """Extract one environment from a batched scene state and move to CPU."""

    def _one(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            return t.detach().cpu().clone()
        if t.shape[0] <= env_id:
            raise IndexError(f"env_id={env_id} out of range for state tensor {tuple(t.shape)}")
        return t[env_id : env_id + 1].detach().cpu().clone()

    return _map_tensors(scene_state, _one)


def expand_single_env_state(scene_state: dict, n: int, device: str | torch.device) -> dict:
    """Expand a single-env snapshot state to exactly ``n`` destination envs."""
    if n <= 0:
        raise ValueError("n must be > 0")

    def _expand(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device)
        if t.ndim == 0:
            return t.clone()
        if t.shape[0] != 1:
            raise ValueError(
                "snapshot tensors must have batch dimension 1; "
                f"got shape {tuple(t.shape)}"
            )
        reps = [n] + [1] * (t.ndim - 1)
        return t.repeat(*reps).clone()

    return _map_tensors(scene_state, _expand)


def randomize_rigid_object_xy(scene_state: dict, asset_name: str, xy_offsets: torch.Tensor) -> None:
    """Add local-frame XY offsets to a rigid object's root pose in-place."""
    try:
        pose = scene_state["rigid_object"][asset_name]["root_pose"]
    except KeyError as exc:
        available = list(scene_state.get("rigid_object", {}).keys())
        raise KeyError(f"rigid object {asset_name!r} missing; available={available}") from exc
    if pose.ndim != 2 or pose.shape[1] < 3:
        raise ValueError(f"invalid root_pose shape for {asset_name}: {tuple(pose.shape)}")
    offsets = torch.as_tensor(xy_offsets, dtype=pose.dtype, device=pose.device)
    if offsets.shape != (pose.shape[0], 2):
        raise ValueError(f"xy_offsets must be {(pose.shape[0], 2)}, got {tuple(offsets.shape)}")
    pose[:, :2] += offsets


def scale_scene_velocities(scene_state: dict, scale: float | torch.Tensor) -> None:
    """Scale saved velocities in-place with one scalar or one factor per environment."""
    factors = torch.as_tensor(scale, dtype=torch.float32)
    if factors.ndim > 1 or not torch.isfinite(factors).all() or not ((0 <= factors) & (factors <= 1)).all():
        raise ValueError("snapshot velocity scale must be a scalar or 1D tensor in [0,1]")
    found = 0
    for group in ("articulation", "rigid_object"):
        for asset in scene_state.get(group, {}).values():
            for name in ("root_velocity", "joint_velocity"):
                value = asset.get(name)
                if value is not None:
                    if not torch.is_tensor(value) or not torch.isfinite(value).all():
                        raise ValueError(f"invalid {group} {name}")
                    factor = factors.to(device=value.device, dtype=value.dtype)
                    if factor.ndim == 1:
                        if len(factor) != value.shape[0]:
                            raise ValueError(
                                f"velocity scale length {len(factor)} does not match batch {value.shape[0]}"
                            )
                        factor = factor.reshape((-1,) + (1,) * (value.ndim - 1))
                    value.mul_(factor)
                    found += 1
    if found == 0:
        raise ValueError("snapshot contains no velocity tensors")


def save_place_snapshot(
    path: str | Path,
    *,
    scene_state_relative: dict,
    task: str,
    source_seed: int,
    trace_path: str,
    place_marker_step: int,
    diagnostics: dict,
    initial_phase: int = 0,
    snapshot_role: str = "place_entry",
    env_id: int = 0,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SNAPSHOT_VERSION,
        "task": str(task),
        "source_seed": int(source_seed),
        "trace_path": str(trace_path),
        "place_marker_step": int(place_marker_step),
        "is_relative": True,
        "initial_phase": int(initial_phase),
        "snapshot_role": str(snapshot_role),
        "scene_state": single_env_cpu_state(scene_state_relative, env_id=env_id),
        "diagnostics": dict(diagnostics),
    }
    torch.save(payload, out)
    return out


def load_place_snapshot(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid snapshot payload type: {type(payload).__name__}")
    if payload.get("version") != SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported snapshot version {payload.get('version')!r}; expected {SNAPSHOT_VERSION!r}"
        )
    if payload.get("is_relative") is not True:
        raise ValueError("M17 PLACE snapshot must be captured with is_relative=True")
    state = payload.get("scene_state")
    if not isinstance(state, dict):
        raise ValueError("snapshot is missing scene_state")
    # Early contract check for the assets used by Stack-Cube.
    if "articulation" not in state or "robot" not in state["articulation"]:
        raise ValueError("snapshot does not contain articulation/robot")
    rigid = state.get("rigid_object", {})
    for name in ("cube_1", "cube_2"):
        if name not in rigid:
            raise ValueError(f"snapshot does not contain rigid_object/{name}")
    phase = int(payload.get("initial_phase", 0))
    if phase < 0 or phase > 4:
        raise ValueError(f"invalid snapshot initial_phase={phase}")
    diagnostics = payload.get("diagnostics", {})
    if diagnostics.get("ready") is not True or diagnostics.get("physical_grasp") is not True:
        raise ValueError("snapshot was not captured from a validated physically-grasped PLACE state")
    return payload
