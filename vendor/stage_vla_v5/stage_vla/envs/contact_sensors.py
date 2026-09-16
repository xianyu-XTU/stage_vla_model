"""Verified per-finger PhysX ContactSensor setup.

M3 functional goal is complete through M3.2-net:
- one net-force ContactSensor per Franka finger;
- object identity is constrained by red-cube fingertip geometry;
- failed filtered/raw contact routes are not active code.

No physical-grasp, reward, stage, or PPO semantics live here.
"""

from __future__ import annotations

LEFT_SENSOR_NAME = "left_finger_contact"
RIGHT_SENSOR_NAME = "right_finger_contact"

LEFT_FINGER_PRIM = "{ENV_REGEX_NS}/Robot/panda_leftfinger"
RIGHT_FINGER_PRIM = "{ENV_REGEX_NS}/Robot/panda_rightfinger"

RED_BLUE_SUPPORT_SENSOR_NAME = "red_blue_support_contact"
RED_CUBE_PRIM = "{ENV_REGEX_NS}/Cube_2"
BLUE_CUBE_PRIM = "{ENV_REGEX_NS}/Cube_1"


def install_finger_net_contact_sensors(
    env_cfg,
    *,
    update_period: float = 0.0,
    history_length: int = 4,
    debug_vis: bool = False,
) -> None:
    """Install the two per-finger ContactSensors verified in the local runtime."""
    if update_period < 0:
        raise ValueError("update_period must be >= 0")
    if history_length < 0:
        raise ValueError("history_length must be >= 0")

    # User-machine M3 verification showed the PhysX-specific cfg is required
    # in this Isaac Lab 3.0 runtime.
    from isaaclab_physx.sensors import ContactSensorCfg

    spawn_cfg = getattr(env_cfg.scene.robot, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "activate_contact_sensors"):
        raise RuntimeError(
            "Robot spawn config does not expose activate_contact_sensors."
        )
    spawn_cfg.activate_contact_sensors = True

    setattr(
        env_cfg.scene,
        LEFT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path=LEFT_FINGER_PRIM,
            update_period=update_period,
            history_length=history_length,
            debug_vis=debug_vis,
        ),
    )
    setattr(
        env_cfg.scene,
        RIGHT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path=RIGHT_FINGER_PRIM,
            update_period=update_period,
            history_length=history_length,
            debug_vis=debug_vis,
        ),
    )


def install_m32_contact_reporting(
    env_cfg,
    *,
    update_period: float = 0.0,
    history_length: int = 4,
    debug_vis: bool = False,
) -> None:
    """Compatibility alias for the final verified M3.2-net sensor setup."""
    install_finger_net_contact_sensors(
        env_cfg,
        update_period=update_period,
        history_length=history_length,
        debug_vis=debug_vis,
    )



def enable_red_blue_contact_reporting(env_cfg) -> None:
    """Enable PhysX ContactReportAPI on both stack cubes before scene build.

    Unlike :func:`install_red_blue_support_contact_sensor`, this function does
    not construct a filtered ContactSensor. It only flips the spawner flags so
    the already-validated synchronous Omni PhysX contact-report reader can
    identify exact Cube_2 <-> Cube_1 contact pairs. This is the default A6-R2
    backend after the filtered sensor caused a Kit/PhysX process-level crash on
    the user's Isaac Sim 6.0.1 / Isaac Lab 3.0 runtime.
    """
    red_cfg = getattr(env_cfg.scene, "cube_2", None)
    blue_cfg = getattr(env_cfg.scene, "cube_1", None)
    if red_cfg is None or blue_cfg is None:
        raise RuntimeError("Expected env_cfg.scene.cube_2 (red) and cube_1 (blue)")
    for name, obj_cfg in (("cube_2", red_cfg), ("cube_1", blue_cfg)):
        spawn_cfg = getattr(obj_cfg, "spawn", None)
        if spawn_cfg is None or not hasattr(spawn_cfg, "activate_contact_sensors"):
            raise RuntimeError(f"{name} spawn config does not expose activate_contact_sensors")
        spawn_cfg.activate_contact_sensors = True

def install_red_blue_support_contact_sensor(
    env_cfg,
    *,
    update_period: float = 0.0,
    history_length: int = 4,
    debug_vis: bool = False,
) -> None:
    """LEGACY/PROBE ONLY: install a filtered red-cube ContactSensor.

    Do not use this as the default M17-A6 R2 backend on the user's runtime: it
    triggered a Kit/PhysX process-level crash. The production R2 path uses
    :func:`enable_red_blue_contact_reporting` plus raw PhysX contact-pair reads.


    Isaac Lab filtered contact reporting is one sensing body to one-or-many
    partner bodies.  Cube_2 (red) is the single sensing body and Cube_1 (blue)
    is the only filter partner, so ``force_matrix_w`` is an exact support-contact
    signal rather than the red cube's total contact force.
    """
    if update_period < 0:
        raise ValueError("update_period must be >= 0")
    if history_length < 0:
        raise ValueError("history_length must be >= 0")

    from isaaclab_physx.sensors import ContactSensorCfg

    red_cfg = getattr(env_cfg.scene, "cube_2", None)
    blue_cfg = getattr(env_cfg.scene, "cube_1", None)
    if red_cfg is None or blue_cfg is None:
        raise RuntimeError("Expected env_cfg.scene.cube_2 (red) and cube_1 (blue)")
    # PhysX contact reporting must be activated on the sensing body. The blue
    # cube is only the filtered partner and does not need its own sensor.
    spawn_cfg = getattr(red_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "activate_contact_sensors"):
        raise RuntimeError("cube_2 spawn config does not expose activate_contact_sensors")
    spawn_cfg.activate_contact_sensors = True

    setattr(
        env_cfg.scene,
        RED_BLUE_SUPPORT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path=RED_CUBE_PRIM,
            update_period=update_period,
            history_length=history_length,
            debug_vis=debug_vis,
            filter_prim_paths_expr=[BLUE_CUBE_PRIM],
        ),
    )
