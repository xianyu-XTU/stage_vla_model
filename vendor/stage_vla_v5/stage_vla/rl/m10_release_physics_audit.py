"""Read-only M10-D4 audit for the physical release chain.

The existing D2 diagnostic answers whether OPEN was selected and whether
geometry/release survived a short transition window.  D4 extends that view
without changing the policy or environment: every post-lift closed->OPEN
command onset is followed for a configurable number of control steps and is
classified from actual finger position, physical grasp, object velocity,
geometry, and settle truth.

This module is pure PyTorch/Python so its event accounting can be tested
without Isaac Lab.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .action_dsl import M9B_OPEN_TOKEN


@dataclass(frozen=True)
class ReleasePhysicsStateBatch:
    """Actual physical/task state at one control boundary."""

    history_ok: Tensor
    geometry_ok: Tensor
    settled: Tensor
    gripper_open: Tensor
    physical_grasp: Tensor
    gripper_width_m: Tensor
    red_linear_speed_mps: Tensor
    red_angular_speed_radps: Tensor

    def validate(self, num_envs: int, device: torch.device) -> "ReleasePhysicsStateBatch":
        bool_fields = (
            "history_ok",
            "geometry_ok",
            "settled",
            "gripper_open",
            "physical_grasp",
        )
        float_fields = (
            "gripper_width_m",
            "red_linear_speed_mps",
            "red_angular_speed_radps",
        )
        for name in (*bool_fields, *float_fields):
            value = torch.as_tensor(getattr(self, name), device=device)
            if value.shape != (num_envs,):
                raise ValueError(
                    f"{name} must have shape ({num_envs},), got {tuple(value.shape)}"
                )
            if name in bool_fields and value.dtype is not torch.bool:
                raise TypeError(f"{name} must be torch.bool")
            if name in float_fields:
                if not value.dtype.is_floating_point:
                    raise TypeError(f"{name} must be floating point")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains NaN/Inf")
        return self

    def released(self) -> Tensor:
        return (
            torch.as_tensor(self.history_ok)
            & torch.as_tensor(self.gripper_open)
            & ~torch.as_tensor(self.physical_grasp)
        )


@dataclass(frozen=True)
class ReleasePhysicsAuditSummary:
    schema: str
    num_envs: int
    steps: int
    horizon: int
    postlift_open_onsets: int
    open_onset_envs: int
    complete_events: int
    incomplete_events: int
    actual_gripper_open_events: int
    actual_release_events: int
    released_on_geometry_events: int
    settled_within_horizon_events: int
    reclose_before_release_events: int
    reclose_after_release_events: int
    physical_regrasp_after_release_events: int
    geometry_lost_before_release_events: int
    geometry_lost_after_release_events: int
    overspeed_after_release_events: int
    released_geometry_never_settled_events: int
    mean_steps_to_actual_open: float
    mean_steps_to_physical_release: float
    mean_steps_to_settled: float
    classification_counts: dict[str, int]


class ReleasePhysicsAuditTracker:
    """Track and classify post-lift closed->OPEN command onsets."""

    schema = "stage_vla.m10_d4.release_physics_audit.v1"

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        horizon: int = 12,
        max_red_linear_speed_mps: float,
        max_red_angular_speed_radps: float,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        if int(horizon) <= 0:
            raise ValueError("horizon must be > 0")
        if max_red_linear_speed_mps < 0 or max_red_angular_speed_radps < 0:
            raise ValueError("speed thresholds must be >= 0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.max_red_linear_speed_mps = float(max_red_linear_speed_mps)
        self.max_red_angular_speed_radps = float(max_red_angular_speed_radps)
        self.steps = 0

        self._events: list[dict[str, object]] = []
        self._pending: list[dict[str, object]] = []
        self._open_onset_seen = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _state_values(self, state: ReleasePhysicsStateBatch, env_id: int) -> dict[str, object]:
        return {
            "history_ok": int(bool(state.history_ok[env_id].item())),
            "geometry": int(bool(state.geometry_ok[env_id].item())),
            "settled": int(bool(state.settled[env_id].item())),
            "gripper_open": int(bool(state.gripper_open[env_id].item())),
            "physical_grasp": int(bool(state.physical_grasp[env_id].item())),
            "released": int(bool(state.released()[env_id].item())),
            "gripper_width_m": float(state.gripper_width_m[env_id].item()),
            "red_linear_speed_mps": float(state.red_linear_speed_mps[env_id].item()),
            "red_angular_speed_radps": float(state.red_angular_speed_radps[env_id].item()),
        }

    def _write_sample(
        self,
        event: dict[str, object],
        *,
        k: int,
        state: ReleasePhysicsStateBatch,
        env_id: int,
        grip_token: Tensor,
        resolved_open: Tensor,
    ) -> None:
        values = self._state_values(state, env_id)
        event[f"k{k}_grip_token"] = int(grip_token[env_id].item())
        event[f"k{k}_resolved_open_command"] = int(bool(resolved_open[env_id].item()))
        for name, value in values.items():
            event[f"k{k}_{name}"] = value

    def _complete_pending(
        self,
        *,
        post: ReleasePhysicsStateBatch,
        grip_token: Tensor,
        resolved_open: Tensor,
    ) -> None:
        remaining: list[dict[str, object]] = []
        for pending in self._pending:
            event = pending["event"]
            env_id = int(pending["env_id"])
            k = int(pending["next_k"])
            self._write_sample(
                event,
                k=k,
                state=post,
                env_id=env_id,
                grip_token=grip_token,
                resolved_open=resolved_open,
            )
            if k >= self.horizon:
                self._classify(event)
            else:
                pending["next_k"] = k + 1
                remaining.append(pending)
        self._pending = remaining

    def _sample_bool(self, event: dict[str, object], k: int, name: str) -> bool:
        return bool(int(event[f"k{k}_{name}"]))

    def _sample_float(self, event: dict[str, object], k: int, name: str) -> float:
        return float(event[f"k{k}_{name}"])

    @staticmethod
    def _first(values: list[bool]) -> int | None:
        for i, value in enumerate(values, start=1):
            if value:
                return i
        return None

    def _classify(self, event: dict[str, object]) -> None:
        ks = list(range(1, self.horizon + 1))
        actual_open = [self._sample_bool(event, k, "gripper_open") for k in ks]
        released = [self._sample_bool(event, k, "released") for k in ks]
        geometry = [self._sample_bool(event, k, "geometry") for k in ks]
        settled = [self._sample_bool(event, k, "settled") for k in ks]
        physical_grasp = [self._sample_bool(event, k, "physical_grasp") for k in ks]
        resolved_open = [
            self._sample_bool(event, k, "resolved_open_command") for k in ks
        ]
        overspeed = [
            self._sample_float(event, k, "red_linear_speed_mps")
            > self.max_red_linear_speed_mps
            or self._sample_float(event, k, "red_angular_speed_radps")
            > self.max_red_angular_speed_radps
            for k in ks
        ]

        open_k = self._first(actual_open)
        release_k = self._first(released)
        settled_k = self._first(
            [released[i] and geometry[i] and settled[i] for i in range(self.horizon)]
        )
        before_release_end = (release_k - 1) if release_k is not None else self.horizon
        reclose_before = any(not resolved_open[i] for i in range(before_release_end))
        reclose_after = (
            release_k is not None
            and any(not resolved_open[i] for i in range(release_k, self.horizon))
        )
        regrasp_after = (
            release_k is not None
            and any(physical_grasp[i] for i in range(release_k, self.horizon))
        )
        geometry_lost_before = (
            release_k is None and any(not value for value in geometry)
        ) or (
            release_k is not None
            and any(not geometry[i] for i in range(release_k - 1))
        )
        geometry_lost_after = (
            release_k is not None
            and any(not geometry[i] for i in range(release_k - 1, self.horizon))
        )
        overspeed_after = (
            release_k is not None
            and any(overspeed[i] for i in range(release_k - 1, self.horizon))
        )
        released_on_geometry = release_k is not None and geometry[release_k - 1]
        released_geometry_never_settled = (
            released_on_geometry and settled_k is None and not geometry_lost_after
        )

        if open_k is None:
            classification = "ACTUATION_NOT_OPEN"
        elif release_k is None and reclose_before:
            classification = "RECLOSE_BEFORE_RELEASE"
        elif release_k is None:
            classification = "PHYSICAL_GRASP_PERSISTS"
        elif not released_on_geometry:
            classification = "RELEASE_OFF_GEOMETRY"
        elif geometry_lost_after:
            classification = "GEOMETRY_LOST_AFTER_RELEASE"
        elif regrasp_after:
            classification = "PHYSICAL_REGRASP_AFTER_RELEASE"
        elif overspeed_after:
            classification = "OVERSPEED_AFTER_RELEASE"
        elif settled_k is None:
            classification = "RELEASED_NOT_SETTLED_WITHIN_HORIZON"
        else:
            classification = "SETTLED_SUCCESS_WITHIN_HORIZON"

        event.update(
            complete=1,
            classification=classification,
            steps_to_actual_open="" if open_k is None else open_k,
            steps_to_physical_release="" if release_k is None else release_k,
            steps_to_settled="" if settled_k is None else settled_k,
            actual_gripper_open=int(open_k is not None),
            actual_release=int(release_k is not None),
            released_on_geometry=int(released_on_geometry),
            settled_within_horizon=int(settled_k is not None),
            reclose_before_release=int(reclose_before),
            reclose_after_release=int(reclose_after),
            physical_regrasp_after_release=int(regrasp_after),
            geometry_lost_before_release=int(geometry_lost_before),
            geometry_lost_after_release=int(geometry_lost_after),
            overspeed_after_release=int(overspeed_after),
            released_geometry_never_settled=int(released_geometry_never_settled),
        )

    def update(
        self,
        *,
        step_index: int,
        token_indices: Tensor,
        pre_gripper_memory_raw: Tensor,
        resolved_gripper_raw: Tensor,
        pre: ReleasePhysicsStateBatch,
        post: ReleasePhysicsStateBatch,
    ) -> None:
        if int(step_index) < 0:
            raise ValueError("step_index must be >= 0")
        pre.validate(self.num_envs, self.device)
        post.validate(self.num_envs, self.device)
        tokens = torch.as_tensor(token_indices, device=self.device)
        memory = torch.as_tensor(pre_gripper_memory_raw, device=self.device)
        resolved = torch.as_tensor(resolved_gripper_raw, device=self.device)
        if tokens.shape != (self.num_envs, 4):
            raise ValueError("token_indices must be [num_envs,4]")
        if memory.shape != (self.num_envs,) or resolved.shape != (self.num_envs,):
            raise ValueError("gripper command tensors must be [num_envs]")

        grip_token = torch.round(tokens[:, 3]).to(torch.long)
        resolved_open = resolved > 0
        self._complete_pending(
            post=post,
            grip_token=grip_token,
            resolved_open=resolved_open,
        )

        onset = (
            torch.as_tensor(pre.history_ok, device=self.device, dtype=torch.bool)
            & (grip_token == M9B_OPEN_TOKEN)
            & (memory < 0)
            & resolved_open
        )
        self._open_onset_seen |= onset
        for env_id in onset.nonzero(as_tuple=False).flatten().tolist():
            pre_values = self._state_values(pre, env_id)
            event: dict[str, object] = {
                "event_step": int(step_index),
                "env_id": int(env_id),
                "complete": 0,
                "classification": "INCOMPLETE_HORIZON",
                "pre_gripper_memory_raw": float(memory[env_id].item()),
                "resolved_gripper_raw": float(resolved[env_id].item()),
            }
            for name, value in pre_values.items():
                event[f"pre_{name}"] = value
            self._write_sample(
                event,
                k=1,
                state=post,
                env_id=env_id,
                grip_token=grip_token,
                resolved_open=resolved_open,
            )
            self._events.append(event)
            if self.horizon == 1:
                self._classify(event)
            else:
                self._pending.append(
                    {"event": event, "env_id": int(env_id), "next_k": 2}
                )
        self.steps += 1

    @staticmethod
    def _mean_nonempty(rows: list[dict[str, object]], field: str) -> float:
        values = [float(row[field]) for row in rows if row.get(field, "") != ""]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> ReleasePhysicsAuditSummary:
        complete = [row for row in self._events if int(row["complete"]) == 1]
        classifications: dict[str, int] = {}
        for row in complete:
            key = str(row["classification"])
            classifications[key] = classifications.get(key, 0) + 1

        def count(field: str) -> int:
            return sum(int(row[field]) for row in complete)

        return ReleasePhysicsAuditSummary(
            schema=self.schema,
            num_envs=self.num_envs,
            steps=self.steps,
            horizon=self.horizon,
            postlift_open_onsets=len(self._events),
            open_onset_envs=int(self._open_onset_seen.sum().item()),
            complete_events=len(complete),
            incomplete_events=len(self._events) - len(complete),
            actual_gripper_open_events=count("actual_gripper_open"),
            actual_release_events=count("actual_release"),
            released_on_geometry_events=count("released_on_geometry"),
            settled_within_horizon_events=count("settled_within_horizon"),
            reclose_before_release_events=count("reclose_before_release"),
            reclose_after_release_events=count("reclose_after_release"),
            physical_regrasp_after_release_events=count("physical_regrasp_after_release"),
            geometry_lost_before_release_events=count("geometry_lost_before_release"),
            geometry_lost_after_release_events=count("geometry_lost_after_release"),
            overspeed_after_release_events=count("overspeed_after_release"),
            released_geometry_never_settled_events=count(
                "released_geometry_never_settled"
            ),
            mean_steps_to_actual_open=self._mean_nonempty(complete, "steps_to_actual_open"),
            mean_steps_to_physical_release=self._mean_nonempty(
                complete, "steps_to_physical_release"
            ),
            mean_steps_to_settled=self._mean_nonempty(complete, "steps_to_settled"),
            classification_counts=classifications,
        )

    @property
    def event_rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._events]


def write_release_physics_json(
    path: str | Path,
    *,
    checkpoint: str,
    project_version: str,
    seed: int,
    summary: ReleasePhysicsAuditSummary,
    rows: Iterable[dict[str, object]],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": summary.schema,
        "checkpoint": str(checkpoint),
        "project_version": str(project_version),
        "seed": int(seed),
        "summary": asdict(summary),
        "events": list(rows),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_release_physics_csv(
    path: str | Path, rows: Iterable[dict[str, object]]
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        out.write_text("", encoding="utf-8")
        return out
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return out
