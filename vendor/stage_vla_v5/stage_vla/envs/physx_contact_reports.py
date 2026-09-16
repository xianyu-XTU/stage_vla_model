"""Direct Omni PhysX contact-report reader for M3.2.

Official Omni PhysX exposes two relevant interfaces:

- ``subscribe_contact_report_events(callback)``: callback after a simulation step
- ``get_contact_report()``: query contact headers/data for the current step

M3.2 deliberately uses ``get_contact_report()`` after each ``env.step()``.
This avoids callback lifetime/order issues and keeps the diagnostic synchronous.

The report header contains actor0/actor1 and collider0/collider1 encoded USD
paths. These are decoded with ``PhysicsSchemaTools.intToSdfPath`` and matched
against the exact Franka finger and red-cube paths.

No experimental Isaac Sim sensor extension is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Any


@dataclass(frozen=True)
class ContactPairEvent:
    """Decoded object-identity and contact diagnostics for one PhysX header."""

    actor0: str
    actor1: str
    collider0: str
    collider1: str
    event_type: int
    num_contact_data: int
    max_impulse: float

    @property
    def has_contact_data(self) -> bool:
        return self.num_contact_data > 0


def _normalize_path(path: str) -> str:
    value = str(path).strip().rstrip("/")
    if not value.startswith("/"):
        value = "/" + value
    return value


def _path_belongs_to(path: str, body_path: str) -> bool:
    path = _normalize_path(path)
    body_path = _normalize_path(body_path)
    return path == body_path or path.startswith(body_path + "/")


def _unordered_match(path0: str, path1: str, body_a: str, body_b: str) -> bool:
    return (
        _path_belongs_to(path0, body_a)
        and _path_belongs_to(path1, body_b)
    ) or (
        _path_belongs_to(path0, body_b)
        and _path_belongs_to(path1, body_a)
    )


def event_matches_body_pair(
    event: ContactPairEvent,
    body_a: str,
    body_b: str,
) -> bool:
    """Whether this header carries actual contact data for body A <-> body B."""
    if not event.has_contact_data:
        return False

    # Actor IDs are the preferred rigid-body identity.
    if _unordered_match(event.actor0, event.actor1, body_a, body_b):
        return True

    # Fallback: collision meshes can live below the rigid-body prim.
    return _unordered_match(event.collider0, event.collider1, body_a, body_b)


def matching_body_pair_events(
    events: Iterable[ContactPairEvent],
    body_a: str,
    body_b: str,
) -> tuple[ContactPairEvent, ...]:
    return tuple(
        event for event in events if event_matches_body_pair(event, body_a, body_b)
    )


def _vec3_magnitude(value: Any) -> float:
    """Magnitude helper supporting tuple-like and carb vector values."""
    try:
        x, y, z = float(value[0]), float(value[1]), float(value[2])
    except Exception:
        x = float(value.x)
        y = float(value.y)
        z = float(value.z)
    return sqrt(x * x + y * y + z * z)


class PhysxContactReportReader:
    """Synchronous reader for Omni PhysX's current-step contact buffer."""

    def __init__(self):
        # Delayed imports keep normal pytest independent of Isaac/Kit.
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._interface = get_physx_simulation_interface()
        if self._interface is None:
            raise RuntimeError("Omni PhysX simulation interface is unavailable.")
        self._PhysicsSchemaTools = PhysicsSchemaTools

    def _decode(self, encoded) -> str:
        return str(self._PhysicsSchemaTools.intToSdfPath(encoded))

    def read_current_step(self) -> tuple[ContactPairEvent, ...]:
        """Read and decode contact-report headers for the current physics step."""
        report = self._interface.get_contact_report()
        if report is None:
            return ()

        if len(report) < 2:
            raise RuntimeError(
                f"Unexpected get_contact_report() return length: {len(report)}"
            )

        contact_headers = report[0]
        contact_data = report[1]

        events: list[ContactPairEvent] = []
        for header in contact_headers:
            offset = int(header.contact_data_offset)
            count = int(header.num_contact_data)

            max_impulse = 0.0
            for index in range(offset, offset + count):
                impulse = contact_data[index].impulse
                max_impulse = max(max_impulse, _vec3_magnitude(impulse))

            events.append(
                ContactPairEvent(
                    actor0=self._decode(header.actor0),
                    actor1=self._decode(header.actor1),
                    collider0=self._decode(header.collider0),
                    collider1=self._decode(header.collider1),
                    event_type=int(header.type),
                    num_contact_data=count,
                    max_impulse=max_impulse,
                )
            )

        return tuple(events)

    def matches(
        self,
        body_a: str,
        body_b: str,
    ) -> tuple[ContactPairEvent, ...]:
        """Read the current step and return exact body-pair contact events."""
        return matching_body_pair_events(
            self.read_current_step(),
            body_a,
            body_b,
        )
