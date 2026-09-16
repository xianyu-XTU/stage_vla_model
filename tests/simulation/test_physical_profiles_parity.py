from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from stage_vla_v7.simulation.isaac_lab.physical_profiles import (
    ISAAC_BLOCK_COLLISION_SIZE_M,
    read_rigid_body_scales,
    set_rigid_body_mass_profile,
    set_rigid_body_scales,
    verify_rigid_body_mass_profile,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.envs.physical_profiles import (  # noqa: E402
    ISAAC_BLOCK_COLLISION_SIZE_M as V5_ISAAC_BLOCK_COLLISION_SIZE_M,
)
from stage_vla.envs.physical_profiles import (  # noqa: E402
    read_rigid_body_scales as v5_read_rigid_body_scales,
)
from stage_vla.envs.physical_profiles import (  # noqa: E402
    set_rigid_body_mass_profile as v5_set_rigid_body_mass_profile,
)
from stage_vla.envs.physical_profiles import (  # noqa: E402
    set_rigid_body_scales as v5_set_rigid_body_scales,
)
from stage_vla.envs.physical_profiles import (  # noqa: E402
    verify_rigid_body_mass_profile as v5_verify_rigid_body_mass_profile,
)


class _Attribute:
    def __init__(self) -> None:
        self.default = None

    def Get(self):
        return self.default


class _PrimSpec:
    def __init__(self, layer, path: str) -> None:
        self.layer = layer
        self.path = path
        self.attributes: dict[str, _Attribute] = {}

    def GetAttributeAtPath(self, path: str):
        return self.attributes.get(str(path))


class _Layer:
    def __init__(self) -> None:
        self.prims: dict[str, _PrimSpec] = {}

    def prim(self, path: str) -> _PrimSpec:
        return self.prims.setdefault(path, _PrimSpec(self, path))


class _RuntimePrim:
    def __init__(self, spec: _PrimSpec) -> None:
        self.spec = spec

    def GetAttribute(self, name: str):
        return self.spec.GetAttributeAtPath(f"{self.spec.path}.{name}")


class _Stage:
    def __init__(self) -> None:
        self.layer = _Layer()

    def GetRootLayer(self):
        return self.layer

    def GetPrimAtPath(self, path: str):
        return _RuntimePrim(self.layer.prim(path))


class _ChangeBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _install_usd_stubs(monkeypatch: pytest.MonkeyPatch, paths: list[str]) -> None:
    isaaclab = ModuleType("isaaclab")
    sim = ModuleType("isaaclab.sim")
    sim.find_matching_prim_paths = lambda _pattern: list(paths)
    isaaclab.sim = sim

    sdf = SimpleNamespace()
    sdf.ChangeBlock = _ChangeBlock
    sdf.ValueTypeNames = SimpleNamespace(Double3="Double3", TokenArray="TokenArray")
    sdf.CreatePrimInLayer = lambda layer, path: layer.prim(str(path))

    def attribute_spec(prim: _PrimSpec, name, _type_name):
        key = str(name)
        if not key.startswith("/"):
            key = f"{prim.path}.{key}"
        attribute = _Attribute()
        prim.attributes[key] = attribute
        return attribute

    sdf.AttributeSpec = attribute_spec
    pxr = ModuleType("pxr")
    pxr.Gf = SimpleNamespace(Vec3f=lambda *values: tuple(float(value) for value in values))
    pxr.Sdf = sdf
    pxr.UsdGeom = SimpleNamespace(Tokens=SimpleNamespace(xformOpOrder="xformOpOrder"))
    pxr.Vt = SimpleNamespace(TokenArray=lambda values: list(values))
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "pxr", pxr)


def _scale_environment(count: int, pattern: str):
    stage = _Stage()
    sim = SimpleNamespace(is_playing=lambda: False, stage=stage)
    asset = SimpleNamespace(cfg=SimpleNamespace(prim_path=pattern))
    scene = {"cube_2": asset}
    scene["num_envs"] = count
    scene = SimpleNamespace(num_envs=count, __getitem__=scene.__getitem__)

    class _Scene:
        num_envs = count

        def __getitem__(self, name):
            return asset

    return SimpleNamespace(sim=sim, scene=_Scene())


class _Asset:
    def __init__(self, masses: torch.Tensor, inertias: torch.Tensor) -> None:
        self.device = torch.device("cpu")
        self.num_instances = masses.shape[0]
        self.data = SimpleNamespace(
            body_mass=masses.clone(),
            body_inertia=inertias.clone(),
        )
        self.mass_write = None
        self.inertia_write = None

    def set_masses_index(self, *, masses, body_ids, env_ids) -> None:
        self.mass_write = (masses.clone(), body_ids.clone(), env_ids.clone())
        self.data.body_mass = masses.clone()

    def set_inertias_index(self, *, inertias, body_ids, env_ids) -> None:
        self.inertia_write = (inertias.clone(), body_ids.clone(), env_ids.clone())
        self.data.body_inertia = inertias.clone()


def test_rigid_body_scale_profile_matches_v5_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [f"/World/envs/env_{index}/Cube_2" for index in range(3)]
    _install_usd_stubs(monkeypatch, paths)
    scales = torch.tensor(
        [[0.8, 0.9, 1.0], [1.0, 1.1, 1.2], [1.2, 1.0, 0.9]],
        dtype=torch.float64,
    )
    actual = _scale_environment(3, "/World/envs/env_.*/Cube_2")
    expected = _scale_environment(3, "/World/envs/env_.*/Cube_2")
    cfg = SimpleNamespace(name="cube_2")

    set_rigid_body_scales(actual, None, scales, cfg)
    v5_set_rigid_body_scales(expected, None, scales, cfg)

    actual_read = read_rigid_body_scales(actual, "cube_2")
    expected_read = v5_read_rigid_body_scales(expected, "cube_2")
    assert actual_read.shape == expected_read.shape == (3, 3)
    assert actual_read.dtype == expected_read.dtype == torch.float32
    assert torch.equal(actual_read, expected_read)
    assert torch.equal(actual_read, scales.float())


def test_mass_and_inertia_profile_matches_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(61081)
    masses = torch.rand((4, 1), generator=generator) + 0.05
    inertias = torch.rand((4, 1, 9), generator=generator) + 0.01
    target = torch.tensor([0.03, 0.05, 0.08, 0.11])
    actual = _Asset(masses, inertias)
    expected = _Asset(masses, inertias)

    set_rigid_body_mass_profile(actual, target)
    v5_set_rigid_body_mass_profile(expected, target)

    assert torch.equal(actual.data.body_mass, expected.data.body_mass)
    assert torch.equal(actual.data.body_inertia, expected.data.body_inertia)
    for actual_args, expected_args in (
        (actual.mass_write, expected.mass_write),
        (actual.inertia_write, expected.inertia_write),
    ):
        assert actual_args is not None and expected_args is not None
        assert all(torch.equal(left, right) for left, right in zip(actual_args, expected_args))
    assert torch.equal(
        verify_rigid_body_mass_profile(actual, target),
        v5_verify_rigid_body_mass_profile(expected, target),
    )


def test_physical_profile_validation_and_constant_match_v5() -> None:
    assert ISAAC_BLOCK_COLLISION_SIZE_M == V5_ISAAC_BLOCK_COLLISION_SIZE_M == 0.047
    asset = _Asset(torch.ones((2, 1)), torch.ones((2, 1, 9)))
    for target in (torch.ones(3), torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 0.0])):
        with pytest.raises((ValueError, RuntimeError)):
            set_rigid_body_mass_profile(asset, target)
    with pytest.raises(RuntimeError, match="simulator mass"):
        verify_rigid_body_mass_profile(asset, torch.tensor([1.0, 2.0]))
