"""Isaac Lab adapters for deterministic per-environment physical profiles."""

from __future__ import annotations

import torch

from .state_reader import to_torch


ISAAC_BLOCK_COLLISION_SIZE_M = 0.047


def set_rigid_body_scales(env, env_ids, scales, asset_cfg) -> None:
    """Author exact USD scales before physics starts."""
    if env.sim.is_playing():
        raise RuntimeError("rigid-body scale profiles must be applied before simulation starts")
    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, UsdGeom, Vt

    values = torch.as_tensor(scales, dtype=torch.float32, device="cpu")
    if values.shape != (env.scene.num_envs, 3):
        raise ValueError("scales must have shape [num_envs,3]")
    if not torch.isfinite(values).all() or torch.any(values <= 0):
        raise ValueError("scales must be positive and finite")
    ids = (
        torch.arange(env.scene.num_envs, device="cpu")
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
    )
    asset = env.scene[asset_cfg.name]
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    if len(prim_paths) != env.scene.num_envs:
        raise RuntimeError("rigid-body prim count does not match the environment count")
    stage = env.sim.stage
    with Sdf.ChangeBlock():
        for env_id in ids.tolist():
            prim_path = prim_paths[env_id]
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(
                    prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3
                )
            scale_spec.default = Gf.Vec3f(*values[env_id].tolist())
            if not has_scale_attr:
                order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if order_spec is None:
                    order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                order_spec.default = Vt.TokenArray(
                    ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                )


def read_rigid_body_scales(env, asset_name: str) -> torch.Tensor:
    """Read authored per-environment USD scales for contract verification."""
    import isaaclab.sim as sim_utils

    asset = env.scene[asset_name]
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    rows: list[tuple[float, float, float]] = []
    for prim_path in prim_paths:
        prim = env.sim.stage.GetPrimAtPath(prim_path)
        scale = prim.GetAttribute("xformOp:scale").Get()
        if scale is None:
            raise RuntimeError(f"missing authored scale for {prim_path}")
        rows.append(tuple(float(value) for value in scale))
    return torch.tensor(rows, dtype=torch.float32)


def set_rigid_body_mass_profile(asset, masses_kg: torch.Tensor) -> None:
    """Set exact masses and scale parsed inertia by the corresponding ratio."""
    target = torch.as_tensor(masses_kg, device=asset.device, dtype=torch.float32)
    if target.shape != (asset.num_instances,):
        raise ValueError("mass profile must have shape [num_instances]")
    if not torch.isfinite(target).all() or torch.any(target <= 0):
        raise ValueError("mass profile must be positive and finite")
    current_mass = to_torch(asset.data.body_mass).clone()
    current_inertia = to_torch(asset.data.body_inertia).clone()
    if current_mass.shape != (asset.num_instances, 1):
        raise RuntimeError("physical-profile assets must contain exactly one rigid body")
    ratio = target / current_mass[:, 0]
    env_ids = torch.arange(asset.num_instances, device=asset.device, dtype=torch.int32)
    body_ids = torch.tensor([0], device=asset.device, dtype=torch.int32)
    asset.set_masses_index(
        masses=target.unsqueeze(-1), body_ids=body_ids, env_ids=env_ids
    )
    asset.set_inertias_index(
        inertias=current_inertia * ratio[:, None, None],
        body_ids=body_ids,
        env_ids=env_ids,
    )


def verify_rigid_body_mass_profile(asset, expected_kg: torch.Tensor) -> torch.Tensor:
    """Return simulator masses after enforcing a tight profile consistency gate."""
    expected = torch.as_tensor(expected_kg, device=asset.device, dtype=torch.float32)
    actual = to_torch(asset.data.body_mass)[:, 0]
    if actual.shape != expected.shape or not torch.allclose(
        actual, expected, atol=1e-6, rtol=1e-5
    ):
        raise RuntimeError("simulator mass does not match the policy physical profile")
    return actual.clone()
