"""Robust red<->blue support-contact provider for M17-A6 R2.

Why this exists
---------------
The first A6-R2 implementation used a filtered Isaac Lab ``ContactSensor`` on
Cube_2. On the user's Isaac Sim 6.0.1 / Isaac Lab 3.0 runtime that path caused
a Kit/PhysX process-level crash before Python could report an exception.

This module deliberately reuses the project's already-validated M3 synchronous
Omni PhysX contact-report path instead. It does not create a ContactSensor.
The cube spawners only need ``activate_contact_sensors=True`` before scene
construction so PhysX emits ContactReportAPI events. Exact red<->blue identity
is then decoded from the current-step contact report.

A6-R2 is evaluation-only, so this provider intentionally supports one env
(``env_0``) at a time. This keeps the held-out 5+20 evaluation protocol intact
and avoids adding a new vectorized sensor dependency to PPO training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .physx_contact_reports import PhysxContactReportReader


RED_BODY_ENV0 = "/World/envs/env_0/Cube_2"
BLUE_BODY_ENV0 = "/World/envs/env_0/Cube_1"


@dataclass
class RawPairSupportDiagnostics:
    matched_events: int = 0
    max_impulse_ns: float = 0.0
    force_estimate_n: float = 0.0


class Env0RawPairSupportProvider:
    """Return a one-element support-force tensor from exact red<->blue pairs.

    ``ContactPairEvent.max_impulse`` is a current-step contact impulse. Dividing
    by the physics step gives a force-like magnitude in Newtons that can use the
    existing A6 ``support_force_threshold_n`` gate. The exact magnitude is also
    logged, but the important semantic signal is that the decoded body pair is
    Cube_2 <-> Cube_1 rather than an aggregate red-cube contact.
    """

    def __init__(self, *, physics_dt: float, device: torch.device | str):
        physics_dt = float(physics_dt)
        if physics_dt <= 0:
            raise ValueError("physics_dt must be > 0")
        self.physics_dt = physics_dt
        self.device = torch.device(device)
        self.reader = PhysxContactReportReader()
        self.last = RawPairSupportDiagnostics()
        self.max_force_n_seen = 0.0
        self.total_matching_events = 0

    def force_per_env(self, *, num_envs: int) -> torch.Tensor:
        if int(num_envs) != 1:
            raise RuntimeError(
                "Env0RawPairSupportProvider is intentionally evaluation-only and "
                f"requires num_envs=1, got {num_envs}"
            )
        events = self.reader.matches(RED_BODY_ENV0, BLUE_BODY_ENV0)
        max_impulse = max((float(ev.max_impulse) for ev in events), default=0.0)
        force_n = max_impulse / self.physics_dt
        self.last = RawPairSupportDiagnostics(
            matched_events=len(events),
            max_impulse_ns=max_impulse,
            force_estimate_n=force_n,
        )
        self.total_matching_events += len(events)
        self.max_force_n_seen = max(self.max_force_n_seen, force_n)
        return torch.tensor([force_n], dtype=torch.float32, device=self.device)
