"""Per-finger PhysX ContactSensor configuration for the V7 runtime."""

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
    if update_period < 0:
        raise ValueError("update_period must be >= 0")
    if history_length < 0:
        raise ValueError("history_length must be >= 0")
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
    install_finger_net_contact_sensors(
        env_cfg,
        update_period=update_period,
        history_length=history_length,
        debug_vis=debug_vis,
    )


def enable_red_blue_contact_reporting(env_cfg) -> None:
    red_cfg = getattr(env_cfg.scene, "cube_2", None)
    blue_cfg = getattr(env_cfg.scene, "cube_1", None)
    if red_cfg is None or blue_cfg is None:
        raise RuntimeError("Expected env_cfg.scene.cube_2 (red) and cube_1 (blue)")
    for name, object_cfg in (("cube_2", red_cfg), ("cube_1", blue_cfg)):
        spawn_cfg = getattr(object_cfg, "spawn", None)
        if spawn_cfg is None or not hasattr(spawn_cfg, "activate_contact_sensors"):
            raise RuntimeError(
                f"{name} spawn config does not expose activate_contact_sensors"
            )
        spawn_cfg.activate_contact_sensors = True


def install_red_blue_support_contact_sensor(
    env_cfg,
    *,
    update_period: float = 0.0,
    history_length: int = 4,
    debug_vis: bool = False,
) -> None:
    """Install the legacy/probe-only filtered red-blue contact sensor."""
    if update_period < 0:
        raise ValueError("update_period must be >= 0")
    if history_length < 0:
        raise ValueError("history_length must be >= 0")
    from isaaclab_physx.sensors import ContactSensorCfg

    red_cfg = getattr(env_cfg.scene, "cube_2", None)
    blue_cfg = getattr(env_cfg.scene, "cube_1", None)
    if red_cfg is None or blue_cfg is None:
        raise RuntimeError("Expected env_cfg.scene.cube_2 (red) and cube_1 (blue)")
    spawn_cfg = getattr(red_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "activate_contact_sensors"):
        raise RuntimeError(
            "cube_2 spawn config does not expose activate_contact_sensors"
        )
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
