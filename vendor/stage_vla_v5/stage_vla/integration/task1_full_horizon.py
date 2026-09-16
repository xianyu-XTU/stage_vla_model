"""Task-1 full-horizon integration helpers.

M19 intentionally does *not* claim an end-to-end learned policy yet.  It is the
first continuous-episode integration benchmark:

    RESET -> validated expert prefix (REACH/GRASP/LIFT/TRANSPORT)
          -> frozen A1_RF150 PLACE controller
          -> strict stable red-on-blue success

The important contract is that the handoff happens in the same live simulator
episode.  No PLACE snapshot restore is allowed in this benchmark, so it can
expose state/physics discontinuities hidden by the M17 PLACE-only protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Task1HandoffDiagnostics:
    physical_grasp: bool
    lifted: bool
    transported_near_goal: bool
    red_local_z_m: float
    red_blue_xy_m: float

    @property
    def ready(self) -> bool:
        return bool(self.physical_grasp and self.lifted and self.transported_near_goal)


def classify_handoff(*, physical_grasp: bool, red_local_z_m: float, red_blue_xy_m: float,
                     lift_height_m: float = 0.05, transport_xy_m: float = 0.15) -> Task1HandoffDiagnostics:
    """Classify the expert->PLACE live handoff with explicit causal milestones."""
    return Task1HandoffDiagnostics(
        physical_grasp=bool(physical_grasp),
        lifted=float(red_local_z_m) > float(lift_height_m),
        transported_near_goal=float(red_blue_xy_m) < float(transport_xy_m),
        red_local_z_m=float(red_local_z_m),
        red_blue_xy_m=float(red_blue_xy_m),
    )


def summarize_task1_rows(rows: Iterable[Mapping]) -> dict:
    """Aggregate full-horizon rows without hiding conditional bottlenecks."""
    rows = list(rows)
    n = len(rows)
    def count(key: str) -> int:
        return sum(bool(r.get(key, False)) for r in rows)

    reach = count("reach_success")
    grasp = count("grasp_success")
    lift = count("lift_success")
    transport = count("transport_success")
    handoff = count("handoff_ready")
    place = count("place_success")
    full = count("full_task_success")
    recovery_attempted = count("recovery_enabled")
    recovery_ready = count("recovery_ready")

    return {
        "total": n,
        "reach_success_count": reach,
        "grasp_success_count": grasp,
        "lift_success_count": lift,
        "transport_success_count": transport,
        "handoff_ready_count": handoff,
        "place_success_count": place,
        "full_task_success_count": full,
        "recovery_attempted_count": recovery_attempted,
        "recovery_ready_count": recovery_ready,
        "full_task_success_rate": (full / n) if n else 0.0,
        "place_given_handoff_rate": (place / handoff) if handoff else 0.0,
        "handoff_rate": (handoff / n) if n else 0.0,
    }
