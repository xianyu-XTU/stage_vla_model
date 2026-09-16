"""Read-only transition diagnostics for the M10 release bottleneck.

D2 intentionally does **not** modify reward, policy, action decoding, stage
transitions, or PPO.  It compares the state that the policy sees before an
action (``s_t``) with the post-physics state used by RewardManager
(``s_{t+1}``) and records how GRIP/XYZ choices interact with release geometry.

The module is pure PyTorch/Python so the core accounting can be regression
-tested without Isaac Lab.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .action_dsl import M9B_CLOSE_TOKEN, M9B_KEEP_TOKEN, M9B_OPEN_TOKEN
from .grip_policy_diagnostics import gripper_probabilities

PLACE_STAGE_ID = 4


@dataclass(frozen=True)
class ReleaseStateBatch:
    """One vectorized physical/task snapshot at a control boundary."""

    stage: Tensor
    geometry_ok: Tensor
    settled: Tensor
    history_ok: Tensor
    gripper_open: Tensor
    physical_grasp: Tensor
    openness: Tensor

    def validate(self, num_envs: int, device: torch.device) -> "ReleaseStateBatch":
        bool_fields = (
            "geometry_ok",
            "settled",
            "history_ok",
            "gripper_open",
            "physical_grasp",
        )
        for name in ("stage", *bool_fields, "openness"):
            value = torch.as_tensor(getattr(self, name), device=device)
            if value.shape != (num_envs,):
                raise ValueError(
                    f"{name} must have shape ({num_envs},), got {tuple(value.shape)}"
                )
            if name in bool_fields and value.dtype is not torch.bool:
                raise TypeError(f"{name} must be torch.bool")
            if name == "stage" and value.dtype.is_floating_point:
                raise TypeError("stage must be an integer tensor")
            if name == "openness":
                if not value.dtype.is_floating_point:
                    raise TypeError("openness must be floating point")
                if not torch.isfinite(value).all():
                    raise ValueError("openness contains NaN/Inf")
        return self

    def release_ready(self, *, place_stage_id: int = PLACE_STAGE_ID) -> Tensor:
        return (
            (torch.as_tensor(self.stage) == int(place_stage_id))
            & torch.as_tensor(self.geometry_ok)
            & torch.as_tensor(self.history_ok)
        )

    def settled_release_ready(self, *, place_stage_id: int = PLACE_STAGE_ID) -> Tensor:
        return self.release_ready(place_stage_id=place_stage_id) & torch.as_tensor(self.settled)

    def released(self) -> Tensor:
        return (
            torch.as_tensor(self.history_ok)
            & torch.as_tensor(self.gripper_open)
            & ~torch.as_tensor(self.physical_grasp)
        )


@dataclass(frozen=True)
class ReleaseTransitionSummary:
    schema: str
    num_envs: int
    steps: int
    horizon: int
    total_samples: int
    pre_ready_samples: int
    post_ready_samples: int
    pre_ready_envs: int
    post_ready_envs: int
    pre_ready_settled_samples: int
    pre_ready_unsettled_samples: int
    pre_ready_unsettled_fraction: float
    entered_ready_samples: int
    exited_ready_samples: int
    open_selected_samples: int
    open_selected_pre_ready: int
    open_selected_post_ready: int
    actual_open_credit_samples: int
    credited_pre_not_ready_post_ready: int
    pre_ready_open_post_not_ready: int
    pre_ready_open_without_credit: int
    post_release_samples: int
    post_release_envs: int
    release_post_geometry: int
    release_post_settled: int
    release_post_geometry_and_settled: int
    release_from_pre_ready: int
    release_from_pre_ready_settled: int
    ready_closed_memory_samples: int
    ready_closed_memory_keep_samples: int
    ready_closed_memory_close_samples: int
    ready_closed_memory_open_samples: int
    mean_p_open_at_pre_ready: float
    mean_p_keep_at_pre_ready: float
    mean_p_close_at_pre_ready: float
    mean_p_physical_open_at_pre_ready: float
    mean_p_physical_closed_at_pre_ready: float
    mean_open_vs_closed_prob_margin_at_pre_ready: float
    open_xyz_nonzero_fraction: float
    credited_open_xyz_nonzero_fraction: float
    open_event_count: int
    retention_complete_open_events: int
    retention_samples_after_open: list[int]
    geometry_retention_after_open: list[float]
    settled_retention_after_open: list[float]
    released_after_open: list[float]
    credited_open_event_count: int
    retention_complete_credited_events: int
    retention_samples_after_credited_open: list[int]
    geometry_retention_after_credited_open: list[float]
    settled_retention_after_credited_open: list[float]
    released_after_credited_open: list[float]


class ReleaseTransitionTracker:
    """Streaming D2 accounting plus event-level future-horizon traces."""

    schema = "stage_vla.m10_d2.release_transition.v1"

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        horizon: int = 4,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        if int(horizon) <= 0:
            raise ValueError("horizon must be > 0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.steps = 0

        self._pre_ready_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_ready_seen = torch.zeros_like(self._pre_ready_seen)
        self._release_seen = torch.zeros_like(self._pre_ready_seen)

        self._counts: dict[str, int] = {}
        self._ready_prob_sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        self._ready_physical_open_prob_sum = 0.0
        self._ready_open_vs_closed_margin_sum = 0.0
        self._ready_prob_samples = 0
        self._open_xyz_nonzero = 0
        self._open_xyz_total = 0
        self._credited_xyz_nonzero = 0
        self._credited_xyz_total = 0

        self._events: list[dict[str, object]] = []
        self._pending: list[dict[str, object]] = []

    def _inc(self, name: str, value: Tensor | int | bool) -> None:
        if torch.is_tensor(value):
            amount = int(torch.as_tensor(value, device=self.device).sum().item())
        else:
            amount = int(value)
        self._counts[name] = self._counts.get(name, 0) + amount

    def _count(self, name: str) -> int:
        return int(self._counts.get(name, 0))

    def _fill_pending_with_post_state(self, post: ReleaseStateBatch) -> None:
        if not self._pending:
            return
        geometry = torch.as_tensor(post.geometry_ok, device=self.device, dtype=torch.bool)
        settled = torch.as_tensor(post.settled, device=self.device, dtype=torch.bool)
        released = post.released().to(device=self.device, dtype=torch.bool)
        still_pending: list[dict[str, object]] = []
        for pending in self._pending:
            row = pending["row"]
            env_id = int(pending["env_id"])
            k = int(pending["next_k"])
            row[f"geometry_after_{k}"] = int(bool(geometry[env_id].item()))
            row[f"settled_after_{k}"] = int(bool(settled[env_id].item()))
            row[f"released_after_{k}"] = int(bool(released[env_id].item()))
            k += 1
            if k <= self.horizon:
                pending["next_k"] = k
                still_pending.append(pending)
        self._pending = still_pending

    def update(
        self,
        *,
        step_index: int,
        actor_logits: Tensor,
        token_indices: Tensor,
        translation_bins: Tensor,
        pre_gripper_memory_raw: Tensor,
        resolved_gripper_raw: Tensor,
        pre: ReleaseStateBatch,
        post: ReleaseStateBatch,
        actual_open_token_credit: Tensor,
        reward_term_post_release_ready: Tensor | None = None,
    ) -> None:
        if int(step_index) < 0:
            raise ValueError("step_index must be >= 0")
        pre.validate(self.num_envs, self.device)
        post.validate(self.num_envs, self.device)

        logits = torch.as_tensor(actor_logits, device=self.device)
        tokens = torch.as_tensor(token_indices, device=self.device)
        bins = torch.as_tensor(translation_bins, device=self.device)
        memory = torch.as_tensor(pre_gripper_memory_raw, device=self.device)
        resolved = torch.as_tensor(resolved_gripper_raw, device=self.device)
        credit = torch.as_tensor(actual_open_token_credit, device=self.device)
        if logits.ndim != 2 or logits.shape[0] != self.num_envs:
            raise ValueError("actor_logits must be [num_envs, total_logits]")
        if tokens.shape != (self.num_envs, 4):
            raise ValueError("token_indices must be [num_envs,4]")
        if bins.shape != (self.num_envs, 3):
            raise ValueError("translation_bins must be [num_envs,3]")
        for name, value in (
            ("pre_gripper_memory_raw", memory),
            ("resolved_gripper_raw", resolved),
            ("actual_open_token_credit", credit),
        ):
            if value.shape != (self.num_envs,):
                raise ValueError(f"{name} must be [num_envs]")
        if not torch.isfinite(credit).all():
            raise ValueError("actual_open_token_credit contains NaN/Inf")

        pre_ready = pre.release_ready().to(device=self.device, dtype=torch.bool)
        post_ready = post.release_ready().to(device=self.device, dtype=torch.bool)
        pre_settled_ready = pre.settled_release_ready().to(device=self.device, dtype=torch.bool)
        post_released = post.released().to(device=self.device, dtype=torch.bool)
        pre_released = pre.released().to(device=self.device, dtype=torch.bool)

        if reward_term_post_release_ready is not None:
            reward_ready = torch.as_tensor(
                reward_term_post_release_ready, device=self.device, dtype=torch.bool
            )
            if reward_ready.shape != (self.num_envs,):
                raise ValueError("reward_term_post_release_ready must be [num_envs]")
            if not torch.equal(reward_ready, post_ready):
                mismatch = int((reward_ready != post_ready).sum().item())
                raise RuntimeError(
                    f"D2 recomputed post release_ready disagrees with reward term in {mismatch} envs"
                )

        grip_token = torch.round(tokens[:, 3]).to(torch.long)
        open_selected = grip_token == M9B_OPEN_TOKEN
        credited = credit > 0
        probs = gripper_probabilities(logits)
        pre_memory_open = memory > 0
        resolved_open = resolved > 0

        # KEEP is an alias for the previous binary command.  Therefore physical
        # OPEN probability depends on decoder memory, not just P(OPEN token).
        p_open = probs[:, M9B_OPEN_TOKEN]
        p_keep = probs[:, M9B_KEEP_TOKEN]
        p_close = probs[:, M9B_CLOSE_TOKEN]
        p_physical_open = torch.where(pre_memory_open, p_open + p_keep, p_open)
        p_physical_closed = 1.0 - p_physical_open

        self._fill_pending_with_post_state(post)

        self._pre_ready_seen |= pre_ready
        self._post_ready_seen |= post_ready
        self._release_seen |= post_released
        self._inc("total_samples", self.num_envs)
        self._inc("pre_ready_samples", pre_ready)
        self._inc("post_ready_samples", post_ready)
        self._inc("pre_ready_settled_samples", pre_settled_ready)
        self._inc("pre_ready_unsettled_samples", pre_ready & ~torch.as_tensor(pre.settled, device=self.device))
        self._inc("entered_ready_samples", ~pre_ready & post_ready)
        self._inc("exited_ready_samples", pre_ready & ~post_ready)
        self._inc("open_selected_samples", open_selected)
        self._inc("open_selected_pre_ready", open_selected & pre_ready)
        self._inc("open_selected_post_ready", open_selected & post_ready)
        self._inc("actual_open_credit_samples", credited)
        self._inc("credited_pre_not_ready_post_ready", credited & ~pre_ready & post_ready)
        self._inc("pre_ready_open_post_not_ready", pre_ready & open_selected & ~post_ready)
        self._inc("pre_ready_open_without_credit", pre_ready & open_selected & ~credited)
        release_onset = post_released & ~pre_released
        self._inc("post_release_samples", release_onset)
        self._inc("release_post_geometry", release_onset & torch.as_tensor(post.geometry_ok, device=self.device))
        self._inc("release_post_settled", release_onset & torch.as_tensor(post.settled, device=self.device))
        self._inc(
            "release_post_geometry_and_settled",
            release_onset
            & torch.as_tensor(post.geometry_ok, device=self.device)
            & torch.as_tensor(post.settled, device=self.device),
        )
        self._inc("release_from_pre_ready", release_onset & pre_ready)
        self._inc("release_from_pre_ready_settled", release_onset & pre_settled_ready)

        ready_closed = pre_ready & ~pre_memory_open
        self._inc("ready_closed_memory_samples", ready_closed)
        self._inc("ready_closed_memory_keep_samples", ready_closed & (grip_token == M9B_KEEP_TOKEN))
        self._inc("ready_closed_memory_close_samples", ready_closed & (grip_token == M9B_CLOSE_TOKEN))
        self._inc("ready_closed_memory_open_samples", ready_closed & open_selected)

        ready_n = int(pre_ready.sum().item())
        if ready_n:
            ready_probs = probs[pre_ready].to(torch.float64)
            self._ready_prob_sums += ready_probs.sum(dim=0)
            phys_open = p_physical_open[pre_ready].to(torch.float64)
            self._ready_physical_open_prob_sum += float(phys_open.sum().item())
            self._ready_open_vs_closed_margin_sum += float(
                (phys_open - p_physical_closed[pre_ready].to(torch.float64)).sum().item()
            )
            self._ready_prob_samples += ready_n

        xyz_nonzero = (bins != 0).any(dim=-1)
        open_n = int(open_selected.sum().item())
        if open_n:
            self._open_xyz_nonzero += int((open_selected & xyz_nonzero).sum().item())
            self._open_xyz_total += open_n
        credited_n = int(credited.sum().item())
        if credited_n:
            self._credited_xyz_nonzero += int((credited & xyz_nonzero).sum().item())
            self._credited_xyz_total += credited_n

        # Event CSV is deliberately sparse: record any explicit OPEN command,
        # any credited OPEN command, or the onset of an actual post-lift release.
        event_mask = open_selected | credited | release_onset
        event_ids = event_mask.nonzero(as_tuple=False).flatten().tolist()
        post_geometry = torch.as_tensor(post.geometry_ok, device=self.device, dtype=torch.bool)
        post_settled = torch.as_tensor(post.settled, device=self.device, dtype=torch.bool)
        for env_id in event_ids:
            row: dict[str, object] = {
                "step": int(step_index),
                "env_id": int(env_id),
                "open_selected": int(bool(open_selected[env_id].item())),
                "open_credited": int(bool(credited[env_id].item())),
                "release_onset": int(bool(release_onset[env_id].item())),
                "open_credit": float(credit[env_id].item()),
                "dx_bin": int(bins[env_id, 0].item()),
                "dy_bin": int(bins[env_id, 1].item()),
                "dz_bin": int(bins[env_id, 2].item()),
                "grip_token": int(grip_token[env_id].item()),
                "pre_gripper_memory_raw": float(memory[env_id].item()),
                "resolved_gripper_raw": float(resolved[env_id].item()),
                "resolved_physical_open_command": int(bool(resolved_open[env_id].item())),
                "p_open_token": float(p_open[env_id].item()),
                "p_keep_token": float(p_keep[env_id].item()),
                "p_close_token": float(p_close[env_id].item()),
                "p_physical_open": float(p_physical_open[env_id].item()),
                "pre_stage": int(torch.as_tensor(pre.stage, device=self.device)[env_id].item()),
                "pre_geometry": int(bool(torch.as_tensor(pre.geometry_ok, device=self.device)[env_id].item())),
                "pre_settled": int(bool(torch.as_tensor(pre.settled, device=self.device)[env_id].item())),
                "pre_history_ok": int(bool(torch.as_tensor(pre.history_ok, device=self.device)[env_id].item())),
                "pre_ready": int(bool(pre_ready[env_id].item())),
                "pre_gripper_open": int(bool(torch.as_tensor(pre.gripper_open, device=self.device)[env_id].item())),
                "pre_physical_grasp": int(bool(torch.as_tensor(pre.physical_grasp, device=self.device)[env_id].item())),
                "pre_openness": float(torch.as_tensor(pre.openness, device=self.device)[env_id].item()),
                "post_stage": int(torch.as_tensor(post.stage, device=self.device)[env_id].item()),
                "post_geometry": int(bool(post_geometry[env_id].item())),
                "post_settled": int(bool(post_settled[env_id].item())),
                "post_history_ok": int(bool(torch.as_tensor(post.history_ok, device=self.device)[env_id].item())),
                "post_ready": int(bool(post_ready[env_id].item())),
                "post_gripper_open": int(bool(torch.as_tensor(post.gripper_open, device=self.device)[env_id].item())),
                "post_physical_grasp": int(bool(torch.as_tensor(post.physical_grasp, device=self.device)[env_id].item())),
                "post_openness": float(torch.as_tensor(post.openness, device=self.device)[env_id].item()),
                "post_released": int(bool(post_released[env_id].item())),
            }
            for k in range(1, self.horizon + 1):
                row[f"geometry_after_{k}"] = ""
                row[f"settled_after_{k}"] = ""
                row[f"released_after_{k}"] = ""
            row["geometry_after_1"] = int(bool(post_geometry[env_id].item()))
            row["settled_after_1"] = int(bool(post_settled[env_id].item()))
            row["released_after_1"] = int(bool(post_released[env_id].item()))
            self._events.append(row)
            if self.horizon >= 2:
                self._pending.append({"row": row, "env_id": int(env_id), "next_k": 2})

        self.steps += 1

    def _retention(self, *, credited_only: bool, field: str) -> tuple[int, list[int], list[float]]:
        selected = [
            row
            for row in self._events
            if int(row["open_selected"]) == 1
            and (not credited_only or int(row["open_credited"]) == 1)
        ]
        complete = 0
        sums = [0 for _ in range(self.horizon)]
        counts = [0 for _ in range(self.horizon)]
        for row in selected:
            all_complete = True
            for k in range(1, self.horizon + 1):
                value = row[f"{field}_after_{k}"]
                if value == "":
                    all_complete = False
                    continue
                counts[k - 1] += 1
                sums[k - 1] += int(value)
            if all_complete:
                complete += 1
        rates = [
            (sums[i] / counts[i]) if counts[i] else 0.0 for i in range(self.horizon)
        ]
        return complete, counts, rates

    def summary(self) -> ReleaseTransitionSummary:
        pre_ready = self._count("pre_ready_samples")
        pre_unsettled = self._count("pre_ready_unsettled_samples")
        if self._ready_prob_samples:
            mean_probs = self._ready_prob_sums / float(self._ready_prob_samples)
            mean_phys_open = self._ready_physical_open_prob_sum / float(self._ready_prob_samples)
            mean_margin = self._ready_open_vs_closed_margin_sum / float(self._ready_prob_samples)
        else:
            mean_probs = torch.zeros(3, dtype=torch.float64, device=self.device)
            mean_phys_open = 0.0
            mean_margin = 0.0

        open_complete_g, open_counts, open_g = self._retention(credited_only=False, field="geometry")
        _, _, open_s = self._retention(credited_only=False, field="settled")
        _, _, open_r = self._retention(credited_only=False, field="released")
        cred_complete_g, cred_counts, cred_g = self._retention(credited_only=True, field="geometry")
        _, _, cred_s = self._retention(credited_only=True, field="settled")
        _, _, cred_r = self._retention(credited_only=True, field="released")

        open_event_count = sum(int(row["open_selected"]) for row in self._events)
        credited_event_count = sum(int(row["open_credited"]) for row in self._events)
        return ReleaseTransitionSummary(
            schema=self.schema,
            num_envs=self.num_envs,
            steps=self.steps,
            horizon=self.horizon,
            total_samples=self._count("total_samples"),
            pre_ready_samples=pre_ready,
            post_ready_samples=self._count("post_ready_samples"),
            pre_ready_envs=int(self._pre_ready_seen.sum().item()),
            post_ready_envs=int(self._post_ready_seen.sum().item()),
            pre_ready_settled_samples=self._count("pre_ready_settled_samples"),
            pre_ready_unsettled_samples=pre_unsettled,
            pre_ready_unsettled_fraction=(pre_unsettled / pre_ready) if pre_ready else 0.0,
            entered_ready_samples=self._count("entered_ready_samples"),
            exited_ready_samples=self._count("exited_ready_samples"),
            open_selected_samples=self._count("open_selected_samples"),
            open_selected_pre_ready=self._count("open_selected_pre_ready"),
            open_selected_post_ready=self._count("open_selected_post_ready"),
            actual_open_credit_samples=self._count("actual_open_credit_samples"),
            credited_pre_not_ready_post_ready=self._count("credited_pre_not_ready_post_ready"),
            pre_ready_open_post_not_ready=self._count("pre_ready_open_post_not_ready"),
            pre_ready_open_without_credit=self._count("pre_ready_open_without_credit"),
            post_release_samples=self._count("post_release_samples"),
            post_release_envs=int(self._release_seen.sum().item()),
            release_post_geometry=self._count("release_post_geometry"),
            release_post_settled=self._count("release_post_settled"),
            release_post_geometry_and_settled=self._count("release_post_geometry_and_settled"),
            release_from_pre_ready=self._count("release_from_pre_ready"),
            release_from_pre_ready_settled=self._count("release_from_pre_ready_settled"),
            ready_closed_memory_samples=self._count("ready_closed_memory_samples"),
            ready_closed_memory_keep_samples=self._count("ready_closed_memory_keep_samples"),
            ready_closed_memory_close_samples=self._count("ready_closed_memory_close_samples"),
            ready_closed_memory_open_samples=self._count("ready_closed_memory_open_samples"),
            mean_p_open_at_pre_ready=float(mean_probs[M9B_OPEN_TOKEN].item()),
            mean_p_keep_at_pre_ready=float(mean_probs[M9B_KEEP_TOKEN].item()),
            mean_p_close_at_pre_ready=float(mean_probs[M9B_CLOSE_TOKEN].item()),
            mean_p_physical_open_at_pre_ready=float(mean_phys_open),
            mean_p_physical_closed_at_pre_ready=float(1.0 - mean_phys_open),
            mean_open_vs_closed_prob_margin_at_pre_ready=float(mean_margin),
            open_xyz_nonzero_fraction=(self._open_xyz_nonzero / self._open_xyz_total)
            if self._open_xyz_total
            else 0.0,
            credited_open_xyz_nonzero_fraction=(
                self._credited_xyz_nonzero / self._credited_xyz_total
            )
            if self._credited_xyz_total
            else 0.0,
            open_event_count=open_event_count,
            retention_complete_open_events=open_complete_g,
            retention_samples_after_open=open_counts,
            geometry_retention_after_open=open_g,
            settled_retention_after_open=open_s,
            released_after_open=open_r,
            credited_open_event_count=credited_event_count,
            retention_complete_credited_events=cred_complete_g,
            retention_samples_after_credited_open=cred_counts,
            geometry_retention_after_credited_open=cred_g,
            settled_retention_after_credited_open=cred_s,
            released_after_credited_open=cred_r,
        )

    @property
    def event_rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._events]


def write_release_transition_json(
    path: str | Path,
    *,
    checkpoint: str,
    project_version: str,
    seed: int,
    summary: ReleaseTransitionSummary,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": summary.schema,
        "checkpoint": str(checkpoint),
        "project_version": str(project_version),
        "seed": int(seed),
        "summary": asdict(summary),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_release_transition_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        out.write_text("", encoding="utf-8")
        return out
    fieldnames = list(materialized[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return out
