"""Pure accounting for M10-D3 reward-scale / terminal-trajectory audit.

D3 is diagnostic-only.  It does not define any training reward.  The tracker
receives already-computed M10 reward components and decoded Action-DSL actions,
then summarizes where H2-S2 settle shaping overlaps GRIP/XYZ decisions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .action_dsl import M9B_CLOSE_TOKEN, M9B_KEEP_TOKEN, M9B_OPEN_TOKEN

D3_SCHEMA = "stage_vla.m10_d3.reward_trajectory.v1"
_EPS = 1.0e-12


def _flat_float(x: Tensor) -> Tensor:
    out = torch.as_tensor(x, dtype=torch.float64).detach().reshape(-1).cpu()
    if out.numel() and not torch.isfinite(out).all():
        raise ValueError("audit values contain NaN/Inf")
    return out


def _flat_bool(x: Tensor) -> Tensor:
    out = torch.as_tensor(x).detach().reshape(-1).cpu()
    if out.dtype is not torch.bool:
        raise TypeError("audit mask must be torch.bool")
    return out


def distribution_stats(values: Tensor) -> dict[str, float | int]:
    """Portable finite-distribution summary used by D3 JSON reports."""
    v = _flat_float(values)
    n = int(v.numel())
    if n == 0:
        return {
            "count": 0,
            "nonzero_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "mean": 0.0,
            "mean_abs": 0.0,
            "p50_abs": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    a = v.abs()
    return {
        "count": n,
        "nonzero_count": int((a > _EPS).sum().item()),
        "positive_count": int((v > _EPS).sum().item()),
        "negative_count": int((v < -_EPS).sum().item()),
        "mean": float(v.mean().item()),
        "mean_abs": float(a.mean().item()),
        "p50_abs": float(torch.quantile(a, 0.50).item()),
        "p95_abs": float(torch.quantile(a, 0.95).item()),
        "max_abs": float(a.max().item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
    }


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    v = _flat_float(values)
    m = _flat_bool(mask)
    if v.shape != m.shape:
        raise ValueError(f"value/mask mismatch: {tuple(v.shape)} vs {tuple(m.shape)}")
    return v[m]


def _token_counts(tokens: Tensor, mask: Tensor | None = None) -> dict[str, int]:
    t = torch.as_tensor(tokens).detach().reshape(-1).to(torch.long).cpu()
    if mask is not None:
        m = _flat_bool(mask)
        if m.shape != t.shape:
            raise ValueError("token/mask shape mismatch")
        t = t[m]
    return {
        "OPEN": int((t == M9B_OPEN_TOKEN).sum().item()),
        "KEEP": int((t == M9B_KEEP_TOKEN).sum().item()),
        "CLOSE": int((t == M9B_CLOSE_TOKEN).sum().item()),
    }


@dataclass(frozen=True)
class D3StepBatch:
    grip_token: Tensor
    translation_bins: Tensor
    xyz_command_norm_m: Tensor
    ee_displacement_m: Tensor
    pre_ready: Tensor
    pre_settled: Tensor
    post_ready: Tensor
    post_settled: Tensor
    post_geometry: Tensor
    base_shaping_reward: Tensor
    transition_bonus: Tensor
    success_bonus: Tensor
    release_credit: Tensor
    open_token_credit: Tensor
    settle_shaping_reward: Tensor
    settle_potential: Tensor
    total_reward: Tensor


class RewardTrajectoryAudit:
    """Streaming D3 collector for deterministic checkpoint evaluation."""

    component_names = (
        "base_shaping_reward",
        "transition_bonus",
        "success_bonus",
        "release_credit",
        "open_token_credit",
        "settle_shaping_reward",
        "settle_potential",
        "total_reward",
        "counterfactual_without_settle",
    )

    def __init__(self, *, num_envs: int) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        self.num_envs = int(num_envs)
        self.steps = 0
        self._data: dict[str, list[Tensor]] = {k: [] for k in self.component_names}
        self._masks: dict[str, list[Tensor]] = {
            "pre_ready": [],
            "pre_settled": [],
            "post_ready": [],
            "post_settled": [],
            "post_geometry": [],
            "settle_active": [],
            "settle_positive": [],
            "open": [],
            "xyz_nonzero": [],
        }
        self._grip_tokens: list[Tensor] = []
        self._xyz_norms: list[Tensor] = []
        self._ee_moves: list[Tensor] = []
        self._events: list[dict[str, object]] = []

    @staticmethod
    def _nvec(name: str, value: Tensor, n: int, *, boolean: bool = False) -> Tensor:
        t = torch.as_tensor(value).detach().reshape(-1).cpu()
        if t.shape != (n,):
            raise ValueError(f"{name} must flatten to ({n},), got {tuple(t.shape)}")
        if boolean:
            if t.dtype is not torch.bool:
                raise TypeError(f"{name} must be bool")
        else:
            t = t.to(torch.float64)
            if not torch.isfinite(t).all():
                raise ValueError(f"{name} contains NaN/Inf")
        return t

    def update(self, *, step_index: int, batch: D3StepBatch) -> None:
        if int(step_index) < 0:
            raise ValueError("step_index must be >= 0")
        n = self.num_envs
        grip = torch.as_tensor(batch.grip_token).detach().reshape(-1).to(torch.long).cpu()
        if grip.shape != (n,):
            raise ValueError("grip_token must be [num_envs]")
        if bool(((grip < 0) | (grip > 2)).any().item()):
            raise ValueError("grip token must be OPEN/KEEP/CLOSE categories 0..2")
        bins = torch.as_tensor(batch.translation_bins).detach().cpu()
        if bins.shape != (n, 3):
            raise ValueError("translation_bins must be [num_envs,3]")

        xyz = self._nvec("xyz_command_norm_m", batch.xyz_command_norm_m, n)
        ee = self._nvec("ee_displacement_m", batch.ee_displacement_m, n)
        pre_ready = self._nvec("pre_ready", batch.pre_ready, n, boolean=True)
        pre_settled = self._nvec("pre_settled", batch.pre_settled, n, boolean=True)
        post_ready = self._nvec("post_ready", batch.post_ready, n, boolean=True)
        post_settled = self._nvec("post_settled", batch.post_settled, n, boolean=True)
        post_geometry = self._nvec("post_geometry", batch.post_geometry, n, boolean=True)

        values: dict[str, Tensor] = {}
        for name in (
            "base_shaping_reward",
            "transition_bonus",
            "success_bonus",
            "release_credit",
            "open_token_credit",
            "settle_shaping_reward",
            "settle_potential",
            "total_reward",
        ):
            values[name] = self._nvec(name, getattr(batch, name), n)
        values["counterfactual_without_settle"] = (
            values["total_reward"] - values["settle_shaping_reward"]
        )

        settle_active = values["settle_shaping_reward"].abs() > _EPS
        settle_positive = values["settle_shaping_reward"] > _EPS
        open_mask = grip == M9B_OPEN_TOKEN
        xyz_nonzero = xyz > 1.0e-9

        for name, v in values.items():
            self._data[name].append(v)
        for name, v in (
            ("pre_ready", pre_ready),
            ("pre_settled", pre_settled),
            ("post_ready", post_ready),
            ("post_settled", post_settled),
            ("post_geometry", post_geometry),
            ("settle_active", settle_active),
            ("settle_positive", settle_positive),
            ("open", open_mask),
            ("xyz_nonzero", xyz_nonzero),
        ):
            self._masks[name].append(v)
        self._grip_tokens.append(grip)
        self._xyz_norms.append(xyz)
        self._ee_moves.append(ee)

        relevant = pre_ready | post_ready | open_mask | settle_active | (values["release_credit"].abs() > _EPS)
        for env_id in relevant.nonzero(as_tuple=False).flatten().tolist():
            self._events.append(
                {
                    "step": int(step_index),
                    "env_id": int(env_id),
                    "grip_token": int(grip[env_id].item()),
                    "dx_bin": int(bins[env_id, 0].item()),
                    "dy_bin": int(bins[env_id, 1].item()),
                    "dz_bin": int(bins[env_id, 2].item()),
                    "xyz_command_norm_mm": float(1000.0 * xyz[env_id].item()),
                    "ee_displacement_mm": float(1000.0 * ee[env_id].item()),
                    "pre_ready": int(pre_ready[env_id].item()),
                    "pre_settled": int(pre_settled[env_id].item()),
                    "post_ready": int(post_ready[env_id].item()),
                    "post_settled": int(post_settled[env_id].item()),
                    "post_geometry": int(post_geometry[env_id].item()),
                    "base_shaping_reward": float(values["base_shaping_reward"][env_id].item()),
                    "release_credit": float(values["release_credit"][env_id].item()),
                    "open_token_credit": float(values["open_token_credit"][env_id].item()),
                    "settle_shaping_reward": float(values["settle_shaping_reward"][env_id].item()),
                    "settle_potential": float(values["settle_potential"][env_id].item()),
                    "total_reward": float(values["total_reward"][env_id].item()),
                    "counterfactual_without_settle": float(
                        values["counterfactual_without_settle"][env_id].item()
                    ),
                }
            )
        self.steps += 1

    def _cat(self, name: str) -> Tensor:
        values = self._data[name]
        return torch.cat(values) if values else torch.empty(0, dtype=torch.float64)

    def _mask(self, name: str) -> Tensor:
        values = self._masks[name]
        return torch.cat(values) if values else torch.empty(0, dtype=torch.bool)

    def summary(self) -> dict[str, object]:
        grip = torch.cat(self._grip_tokens) if self._grip_tokens else torch.empty(0, dtype=torch.long)
        xyz = torch.cat(self._xyz_norms) if self._xyz_norms else torch.empty(0, dtype=torch.float64)
        ee = torch.cat(self._ee_moves) if self._ee_moves else torch.empty(0, dtype=torch.float64)
        masks = {k: self._mask(k) for k in self._masks}
        total_samples = int(grip.numel())

        components: dict[str, object] = {}
        for name in self.component_names:
            v = self._cat(name)
            components[name] = {
                "all": distribution_stats(v),
                "pre_ready": distribution_stats(_masked(v, masks["pre_ready"])),
                "settle_active": distribution_stats(_masked(v, masks["settle_active"])),
            }

        settle = self._cat("settle_shaping_reward")
        stage = self._cat("base_shaping_reward")
        release = self._cat("release_credit")
        total = self._cat("total_reward")
        cf = self._cat("counterfactual_without_settle")
        active = masks["settle_active"]
        positive = masks["settle_positive"]
        open_mask = masks["open"]
        pre_ready = masks["pre_ready"]
        pre_ready_open = pre_ready & open_mask

        def frac(mask: Tensor, denom_mask: Tensor | None = None) -> float:
            if denom_mask is None:
                denom = int(mask.numel())
                return float(mask.sum().item() / denom) if denom else 0.0
            denom = int(denom_mask.sum().item())
            return float((mask & denom_mask).sum().item() / denom) if denom else 0.0

        if bool(active.any().item()):
            denom = stage[active].abs() + release[active].abs() + 1.0e-12
            settle_vs_stage_release = float((settle[active].abs() / denom).mean().item())
        else:
            settle_vs_stage_release = 0.0

        sign_flip = ((cf > _EPS) & (total < -_EPS)) | ((cf < -_EPS) & (total > _EPS))
        action = {
            "grip_counts_all": _token_counts(grip),
            "grip_counts_pre_ready": _token_counts(grip, pre_ready),
            "grip_counts_settle_positive": _token_counts(grip, positive),
            "xyz_command_norm_m_all": distribution_stats(xyz),
            "xyz_command_norm_m_pre_ready": distribution_stats(_masked(xyz, pre_ready)),
            "xyz_command_norm_m_pre_ready_open": distribution_stats(_masked(xyz, pre_ready_open)),
            "xyz_command_norm_m_settle_positive": distribution_stats(_masked(xyz, positive)),
            "ee_displacement_m_all": distribution_stats(ee),
            "ee_displacement_m_pre_ready": distribution_stats(_masked(ee, pre_ready)),
            "ee_displacement_m_pre_ready_open": distribution_stats(_masked(ee, pre_ready_open)),
            "ee_displacement_m_settle_positive": distribution_stats(_masked(ee, positive)),
            "pre_ready_xyz_nonzero_fraction": frac(masks["xyz_nonzero"], pre_ready),
            "pre_ready_open_xyz_nonzero_fraction": frac(masks["xyz_nonzero"], pre_ready_open),
            "settle_positive_xyz_nonzero_fraction": frac(masks["xyz_nonzero"], positive),
            "settle_positive_open_fraction": frac(open_mask, positive),
        }
        terminal = {
            "pre_ready_samples": int(pre_ready.sum().item()),
            "pre_ready_settled_samples": int((pre_ready & masks["pre_settled"]).sum().item()),
            "post_ready_samples": int(masks["post_ready"].sum().item()),
            "post_settled_samples": int(masks["post_settled"].sum().item()),
            "pre_ready_open_samples": int(pre_ready_open.sum().item()),
            "pre_ready_open_post_geometry": int(
                (pre_ready_open & masks["post_geometry"]).sum().item()
            ),
            "pre_ready_open_post_settled": int(
                (pre_ready_open & masks["post_settled"]).sum().item()
            ),
        }
        competition = {
            "settle_active_samples": int(active.sum().item()),
            "settle_positive_samples": int(positive.sum().item()),
            "settle_positive_with_open_token": int((positive & open_mask).sum().item()),
            "settle_positive_with_nonzero_xyz": int(
                (positive & masks["xyz_nonzero"]).sum().item()
            ),
            "mean_abs_settle_over_abs_stage_plus_release_when_active": settle_vs_stage_release,
            "reward_sign_flip_due_to_settle_count": int(sign_flip.sum().item()),
        }
        return {
            "schema": D3_SCHEMA,
            "num_envs": self.num_envs,
            "steps": self.steps,
            "total_samples": total_samples,
            "components": components,
            "action": action,
            "terminal": terminal,
            "competition": competition,
        }

    @property
    def event_rows(self) -> list[dict[str, object]]:
        return list(self._events)


def write_d3_json(path: str | Path, payload: dict[str, object]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_d3_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
