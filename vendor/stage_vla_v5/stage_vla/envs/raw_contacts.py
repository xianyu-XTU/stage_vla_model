"""Isaac Sim 6.0.1 raw contact-pair sensor bridge for M3.1.

The old ``isaacsim.sensors.physics`` API is deprecated in Isaac Sim 6.0.
This module uses ``isaacsim.sensors.experimental.physics.ContactSensor``.

M3.1 is intentionally a single-environment diagnostic. The raw Python contact
records are used as object-identity ground truth, not yet as the final 128-env
training observation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .isaac_extensions import ensure_physics_sensor_extension


@dataclass
class RawContactSensorSet:
    """Runtime raw sensors scoped to the two fingers and red cube."""

    finger_a: Any
    finger_b: Any
    red_cube: Any

    finger_a_parent: str
    finger_b_parent: str
    red_cube_parent: str


def _assert_prim(stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Required rigid-body prim does not exist: {path}")


def _create_raw_sensor(parent_path: str, sensor_name: str):
    # AppLauncher experiences do not necessarily enable every Isaac Sim
    # extension. M3.1-R1 explicitly enables/discovers this one first.
    project_root = Path(__file__).resolve().parents[2]
    ensure_physics_sensor_extension(project_root)

    # Delayed import keeps normal pytest independent of Isaac Sim.
    try:
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Physics sensor extension reports enabled, but its Python module "
            "still cannot be imported. Check the configured Isaac Sim root "
            "and extension version."
        ) from exc

    sensor_path = f"{parent_path}/{sensor_name}"
    contact = Contact.create(
        sensor_path,
        min_threshold=0.0,
        max_threshold=1.0e9,
        translations=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    return ContactSensor(contact)


def create_env0_raw_contact_sensors() -> RawContactSensorSet:
    """Create three raw-contact sensors for the official env_0 stack scene.

    Paths are deliberately explicit because M3.1 is a one-environment diagnostic.
    The official task currently uses the ``/World/envs/env_0`` clone namespace.

    The two finger sensors are named A/B rather than relying on semantic
    left/right interpretation. The parent USD body path is still recorded.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    finger_a_parent = "/World/envs/env_0/Robot/panda_leftfinger"
    finger_b_parent = "/World/envs/env_0/Robot/panda_rightfinger"
    red_cube_parent = "/World/envs/env_0/Cube_2"

    for path in (finger_a_parent, finger_b_parent, red_cube_parent):
        _assert_prim(stage, path)

    return RawContactSensorSet(
        finger_a=_create_raw_sensor(finger_a_parent, "StageVLA_RawContact_A"),
        finger_b=_create_raw_sensor(finger_b_parent, "StageVLA_RawContact_B"),
        red_cube=_create_raw_sensor(red_cube_parent, "StageVLA_RawContact_Red"),
        finger_a_parent=finger_a_parent,
        finger_b_parent=finger_b_parent,
        red_cube_parent=red_cube_parent,
    )
