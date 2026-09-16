"""Reset-time helpers for deterministic rigid-object positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _xyz(
    value: Sequence[float], *, name: str, device: torch.device
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.ndim not in (1, 2) or result.shape[-1] != 3:
        raise ValueError(
            f"{name} must contain [x, y, z] or a batch with shape [N, 3]"
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _selected_xyz_rows(
    value: torch.Tensor,
    *,
    ids: torch.Tensor,
    num_envs: int,
    name: str,
) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(0).expand(len(ids), -1)
    if value.shape[0] == int(num_envs):
        return value[ids]
    if value.shape[0] == len(ids):
        return value
    raise ValueError(
        f"{name} batch must have one row per selected or total environment"
    )


def set_fixed_pair_pose(
    env,
    env_ids: torch.Tensor,
    *,
    blue_xyz: Sequence[float],
    red_xyz: Sequence[float],
    blue_name: str = "cube_1",
    red_name: str = "cube_2",
    min_xy_separation_m: float = 0.045,
) -> None:
    if env_ids is None or len(env_ids) == 0:
        return
    if float(min_xy_separation_m) < 0.0:
        raise ValueError("min_xy_separation_m must be non-negative")
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    blue = _xyz(blue_xyz, name="blue_xyz", device=env.device)
    red = _xyz(red_xyz, name="red_xyz", device=env.device)
    if blue.ndim != red.ndim or (blue.ndim == 2 and blue.shape != red.shape):
        raise ValueError("blue_xyz and red_xyz must have matching shapes")
    if blue.ndim == 2 and blue.shape[0] not in (len(ids), env.num_envs):
        raise ValueError(
            "batched fixed positions must have one row per selected or total environment"
        )
    xy_distance = torch.linalg.vector_norm(
        blue[..., :2] - red[..., :2], dim=-1
    )
    too_close = xy_distance < float(min_xy_separation_m)
    if bool(too_close.any()):
        bad = int(too_close.flatten().nonzero()[0].item())
        distance = float(xy_distance.flatten()[bad])
        raise ValueError(
            "blue_xyz and red_xyz are too close in XY at "
            f"row {bad}: {distance:.4f} m < {float(min_xy_separation_m):.4f} m"
        )
    origins = env.scene.env_origins[ids, :3]
    for name, local_xyz in ((blue_name, blue), (red_name, red)):
        asset = env.scene[name]
        positions = _selected_xyz_rows(
            local_xyz, ids=ids, num_envs=env.num_envs, name=name
        ) + origins
        orientations = asset.data.root_quat_w[ids].clone()
        asset.write_root_pose_to_sim_index(
            root_pose=torch.cat([positions, orientations], dim=-1),
            env_ids=ids,
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(ids), 6, device=env.device),
            env_ids=ids,
        )


def set_fixed_asset_poses(
    env,
    env_ids: torch.Tensor,
    *,
    asset_xyz: Mapping[str, Sequence[float]],
    min_xy_separation_m: float = 0.08,
) -> None:
    if env_ids is None or len(env_ids) == 0:
        return
    if len(asset_xyz) < 2:
        raise ValueError("asset_xyz must contain at least two assets")
    if float(min_xy_separation_m) < 0.0:
        raise ValueError("min_xy_separation_m must be non-negative")
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    positions = {
        str(name): _xyz(value, name=f"asset_xyz[{name!r}]", device=env.device)
        for name, value in asset_xyz.items()
    }
    selected = {
        name: _selected_xyz_rows(
            value,
            ids=ids,
            num_envs=env.num_envs,
            name=f"asset_xyz[{name!r}]",
        )
        for name, value in positions.items()
    }
    names = tuple(positions)
    for index, name in enumerate(names):
        try:
            env.scene[name]
        except KeyError:
            raise ValueError(f"unknown scene asset: {name!r}")
        for other in names[index + 1 :]:
            distance = torch.linalg.vector_norm(
                selected[name][..., :2] - selected[other][..., :2], dim=-1
            )
            too_close = distance < float(min_xy_separation_m)
            if bool(too_close.any()):
                bad = int(too_close.nonzero()[0].item())
                raise ValueError(
                    f"assets {name!r} and {other!r} are too close in XY at row {bad}: "
                    f"{float(distance[bad]):.4f} m < {float(min_xy_separation_m):.4f} m"
                )
    origins = env.scene.env_origins[ids, :3]
    for name, local_xyz in selected.items():
        asset = env.scene[name]
        world_xyz = local_xyz + origins
        orientations = asset.data.root_quat_w[ids].clone()
        asset.write_root_pose_to_sim_index(
            root_pose=torch.cat([world_xyz, orientations], dim=-1),
            env_ids=ids,
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(ids), 6, device=env.device),
            env_ids=ids,
        )
