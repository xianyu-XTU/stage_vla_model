"""Source-focused ALIGN-v2 curriculum that retains anchor-source coverage."""
from __future__ import annotations

import torch

from .place_snapshot import expand_single_env_state, randomize_rigid_object_xy
from .stageppo_align_core import align_errors
from .stageppo_align_v2_core import align_v2_entry_geometry
from .stageppo_align_v2_vecenv import AlignV2VecEnv


def focused_source_choices(
    count: int,
    source_seeds: list[int],
    focus_source_seed: int,
    focus_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1 or len(source_seeds) < 2 or len(set(source_seeds)) != len(source_seeds):
        raise ValueError("invalid source allocation inputs")
    if focus_source_seed not in source_seeds or not 0 < focus_fraction < 1:
        raise ValueError("invalid focus source or fraction")
    focus_index = source_seeds.index(focus_source_seed)
    anchors = [index for index in range(len(source_seeds)) if index != focus_index]
    focus_count = max(1, min(count - 1, round(count * focus_fraction))) if count > 1 else 1
    anchor_count = count - focus_count
    choices = [focus_index] * focus_count
    choices.extend(anchors[index % len(anchors)] for index in range(anchor_count))
    values = torch.tensor(choices, dtype=torch.long)
    return values[torch.randperm(count, generator=generator)]


class FocusedAlignV2VecEnv(AlignV2VecEnv):
    def __init__(
        self,
        env,
        *,
        snapshots,
        cfg=None,
        xy_jitter_m=0.025,
        seed=38,
        focus_source_seed=1015,
        focus_fraction=0.5,
    ):
        super().__init__(env, snapshots=snapshots, cfg=cfg, xy_jitter_m=xy_jitter_m, seed=seed)
        self.focus_source_seed = int(focus_source_seed)
        self.focus_fraction = float(focus_fraction)
        self.source_seeds = [int(payload["source_seed"]) for payload in self.payloads]
        if self.focus_source_seed not in self.source_seeds or not 0 < self.focus_fraction < 1:
            raise ValueError("invalid focused ALIGN curriculum")
        self.interface = {
            **self.interface,
            "training_source_curriculum": {
                "focus_source_seed": self.focus_source_seed,
                "focus_fraction": self.focus_fraction,
                "anchor_sources": [seed for seed in self.source_seeds if seed != self.focus_source_seed],
            },
        }

    def _restore(self, ids):
        if not len(ids):
            return
        self.restore_calls += 1
        choices = focused_source_choices(
            len(ids), self.source_seeds, self.focus_source_seed, self.focus_fraction, self.rng
        )
        offsets = (torch.rand(len(ids), 2, generator=self.rng) * 2 - 1) * self.xy_jitter_m
        for choice in choices.unique().tolist():
            mask = choices == choice
            selected = ids[mask.to(self.device)]
            state = expand_single_env_state(
                self.payloads[choice]["scene_state"], len(selected), self.device
            )
            delta = offsets[mask].to(self.device)
            randomize_rigid_object_xy(state, "cube_1", delta)
            self.unwrapped.reset_to(state, env_ids=selected, is_relative=True)
            self.last_source_idx[selected] = choice
            self.last_offsets[selected] = delta
        self.steps[ids] = self.stable_count[ids] = self.bad_count[ids] = 0
        self.prev_unit[ids] = 0
        self.returns[ids] = 0
        self.finished[ids] = False
        self.unwrapped.episode_length_buf[ids] = 0
        measured = self._measure()
        geometry = align_v2_entry_geometry(measured)
        xy, height, _ = align_errors(measured, self.task_cfg)
        valid = (
            geometry
            & (height > self.task_cfg.height_failure_m)
            & (height < 0.15)
            & (xy < self.task_cfg.far_xy_m)
        )
        if not valid[ids].all():
            failed = ids[~valid[ids]].tolist()
            raise RuntimeError(f"invalid focused ALIGN-v2 restored entries: {failed}")
