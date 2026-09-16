"""Runtime Isaac Sim extension activation helpers.

M3.1 failed because ``isaacsim.sensors.experimental.physics`` exists on disk
but was not enabled by the AppLauncher experience, so its Python namespace was
never registered.

This module fixes only extension discovery/activation. It does not change the
M3.1 contact-identity acceptance standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stage_vla.core import resolve_isaac_sim_root

PHYSICS_SENSOR_EXTENSION = "isaacsim.sensors.experimental.physics"


@dataclass(frozen=True)
class ExtensionActivationReport:
    extension_id: str
    enabled_before: bool
    enabled_after: bool
    immediate_enable_succeeded: bool
    added_search_path: str | None
    enabled_extension_id: str | None


def _try_enable(manager, extension_id: str) -> bool:
    """Enable an already-discoverable Kit extension immediately."""
    try:
        return bool(manager.set_extension_enabled_immediate(extension_id, True))
    except Exception as exc:
        raise RuntimeError(
            f"Kit ExtensionManager failed while enabling {extension_id!r}: {exc}"
        ) from exc


def ensure_physics_sensor_extension(project_root: Path) -> ExtensionActivationReport:
    """Ensure Isaac Sim 6.x experimental physics-sensor extension is enabled.

    The function first tries the current Kit search paths. If the extension is
    not discoverable/enabled, it adds the configured standalone Isaac Sim
    ``exts`` directory to the ExtensionManager search paths and retries.

    This function must be called only *after* AppLauncher/SimulationApp exists.
    """
    import omni.kit.app

    extension_id = PHYSICS_SENSOR_EXTENSION
    app = omni.kit.app.get_app()
    if app is None:
        raise RuntimeError(
            "Kit app is not running. Call ensure_physics_sensor_extension() "
            "only after AppLauncher has created the application."
        )

    manager = app.get_extension_manager()
    enabled_before = bool(manager.is_extension_enabled(extension_id))
    if enabled_before:
        enabled_id = manager.get_enabled_extension_id(extension_id)
        return ExtensionActivationReport(
            extension_id=extension_id,
            enabled_before=True,
            enabled_after=True,
            immediate_enable_succeeded=True,
            added_search_path=None,
            enabled_extension_id=str(enabled_id) if enabled_id else None,
        )

    # First attempt: the extension may already be discoverable but disabled.
    immediate_ok = _try_enable(manager, extension_id)
    if manager.is_extension_enabled(extension_id):
        enabled_id = manager.get_enabled_extension_id(extension_id)
        return ExtensionActivationReport(
            extension_id=extension_id,
            enabled_before=False,
            enabled_after=True,
            immediate_enable_succeeded=immediate_ok,
            added_search_path=None,
            enabled_extension_id=str(enabled_id) if enabled_id else None,
        )

    # Fallback for this project's separate Isaac Sim standalone installation:
    # explicitly register its extension collection directory.
    isaac_sim_root = resolve_isaac_sim_root(project_root).expanduser().resolve()
    exts_root = isaac_sim_root / "exts"
    expected_ext_dir = exts_root / extension_id

    if not exts_root.is_dir():
        raise RuntimeError(
            "Could not enable physics sensor extension and configured Isaac Sim "
            f"extension root does not exist: {exts_root}"
        )
    if not expected_ext_dir.is_dir():
        raise RuntimeError(
            "Configured Isaac Sim root exists, but the required extension folder "
            f"was not found: {expected_ext_dir}"
        )

    try:
        manager.add_path(str(exts_root))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to add Isaac Sim extension search path {exts_root}: {exc}"
        ) from exc

    # Apply any extension-manager changes before solving/enabling again.
    try:
        manager.process_and_apply_all_changes()
    except Exception:
        # add_path + immediate enable is normally sufficient; this method is a
        # best-effort flush and is not required by the acceptance test.
        pass

    immediate_ok = _try_enable(manager, extension_id)
    enabled_after = bool(manager.is_extension_enabled(extension_id))
    enabled_id = manager.get_enabled_extension_id(extension_id)

    if not enabled_after:
        # Include the dependency solver's diagnostic when available.
        solver_text = ""
        try:
            solved, solution, error = manager.solve_extensions(
                [extension_id],
                add_enabled=True,
                return_only_disabled=False,
            )
            solver_text = (
                f" solve_extensions: solved={solved}, solution={list(solution)}, "
                f"error={error!r}."
            )
        except Exception as exc:
            solver_text = f" solve_extensions diagnostic failed: {exc!r}."

        raise RuntimeError(
            "Isaac Sim physics sensor extension is still disabled after adding "
            f"{exts_root} and enabling {extension_id!r}.{solver_text}"
        )

    return ExtensionActivationReport(
        extension_id=extension_id,
        enabled_before=False,
        enabled_after=True,
        immediate_enable_succeeded=immediate_ok,
        added_search_path=str(exts_root),
        enabled_extension_id=str(enabled_id) if enabled_id else None,
    )
